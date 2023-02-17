// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"

namespace mlir {
namespace concretelang {
namespace TFHE {

GLWESecretKey::GLWESecretKey() {
  dimension = -1;
  polySize = -1;
  id = -1;
}

GLWESecretKey::GLWESecretKey(int64_t dimension, int64_t polySize, int64_t id) {
  assert(dimension > 0);
  assert(polySize > 0);
  assert(id > 0);
  this->dimension = dimension;
  this->polySize = polySize;
  this->id = id;
}

bool GLWESecretKey::operator==(GLWESecretKey other) {
  return this->id == other.id && this->dimension == other.dimension &&
         this->polySize == other.polySize;
}

bool GLWESecretKey::operator==(const GLWESecretKey other) const {
  return this->id == other.id && this->dimension == other.dimension &&
         this->polySize == other.polySize;
}

bool GLWESecretKey::operator!=(GLWESecretKey other) {
  return this->id != other.id || this->dimension != other.dimension ||
         this->polySize != other.polySize;
}

std::optional<int64_t> GLWESecretKey::getDimension() const {
  if (this->isNotParameterized()) {
    return std::nullopt;
  } else {
    return this->dimension;
  }
}

std::optional<int64_t> GLWESecretKey::getPolySize() const {
  if (this->isNotParameterized()) {
    return std::nullopt;
  } else {
    return this->polySize;
  }
}

mlir::Optional<int64_t> GLWESecretKey::getId() const {
  if (this->isNotParameterized()) {
    return std::nullopt;
  } else {
    return this->id;
  }
}

bool GLWESecretKey::isNotParameterized() const { return id <= 0; }

llvm::hash_code hash_value(const GLWESecretKey &key) {
  return llvm::hash_combine("GlweSecretKey", key.getDimension(),
                            key.getPolySize(), key.getId());
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir

namespace mlir {
AsmPrinter &operator<<(AsmPrinter &p,
                       mlir::concretelang::TFHE::GLWESecretKey key) {
  if (key.isNotParameterized()) {
    p << "sk[?]";
  } else {
    p << "sk[" << key.getId() << "]<" << key.getPolySize().value() << ","
      << key.getDimension().value() << ">";
  }
  return p;
}

FailureOr<mlir::concretelang::TFHE::GLWESecretKey>
FieldParser<mlir::concretelang::TFHE::GLWESecretKey>::parse(AsmParser &parser) {
  int64_t dimension = -1, polySize = -1, id = -1;
  if (parser.parseKeyword("sk") || parser.parseLSquare()) {
    return mlir::failure();
  }
  auto maybeId = parser.parseOptionalInteger(id);
  if (maybeId.has_value()) {
    if (maybeId.value() || parser.parseRSquare() || parser.parseLess() ||
        parser.parseInteger(polySize) || parser.parseComma() ||
        parser.parseInteger(dimension) || parser.parseGreater()) {
      return mlir::failure();
    }
  } else {
    if (parser.parseQuestion() || parser.parseRSquare()) {
      return mlir::failure();
    }
  }
  if (id <= 0) {
    return mlir::concretelang::TFHE::GLWESecretKey();
  } else {
    return mlir::concretelang::TFHE::GLWESecretKey(dimension, polySize, id);
  }
}
} // namespace mlir
