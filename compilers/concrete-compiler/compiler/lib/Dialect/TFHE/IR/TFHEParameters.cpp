// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"
#include "concretelang/Support/Variants.h"
#include "mlir/IR/OpImplementation.h"
#include <variant>

namespace mlir {
namespace concretelang {
namespace TFHE {

bool GLWESecretKeyParameterized::operator==(
    const GLWESecretKeyParameterized other) const {
  return this->dimension == other.dimension &&
         this->identifier == other.identifier &&
         this->polySize == other.polySize;
}

bool GLWESecretKeyNormalized::operator==(
    const GLWESecretKeyNormalized other) const {
  return this->dimension == other.dimension && this->index == other.index &&
         this->polySize == other.polySize;
}

GLWESecretKey GLWESecretKey::newNone() {
  return GLWESecretKey{GLWESecretKeyNone{}};
}

GLWESecretKey GLWESecretKey::newNormalized(uint64_t dimension,
                                           uint64_t polySize, uint64_t index) {
  return GLWESecretKey{GLWESecretKeyNormalized{dimension, polySize, index}};
}

GLWESecretKey GLWESecretKey::newParameterized(uint64_t dimension,
                                              uint64_t polySize,
                                              uint64_t identifier) {
  return GLWESecretKey{
      GLWESecretKeyParameterized{dimension, polySize, identifier}};
}

bool GLWESecretKey::operator==(const GLWESecretKey other) const {
  return std::visit(
      overloaded{
          [](GLWESecretKeyNone thisK, GLWESecretKeyNone otherK) {
            return true;
          },
          [](GLWESecretKeyNormalized thisK, GLWESecretKeyNormalized otherK) {
            return thisK == otherK;
          },
          [](GLWESecretKeyParameterized thisK,
             GLWESecretKeyParameterized otherK) { return thisK == otherK; },
          [](auto _thisK, auto _otherK) { return false; }},
      this->inner, other.inner);
}

bool GLWESecretKey::operator!=(const GLWESecretKey other) const {
  return !(*this == other);
}

template <typename V> bool GLWESecretKey::is() {
  return std::holds_alternative<V>(this->inner);
}

bool GLWESecretKey::isNone() { return is<GLWESecretKeyNone>(); }
bool GLWESecretKey::isParameterized() {
  return is<GLWESecretKeyParameterized>();
}
bool GLWESecretKey::isNormalized() { return is<GLWESecretKeyNormalized>(); }

template <typename V> std::optional<V> GLWESecretKey::get() {
  if (this->is<V>()) {
    return std::get<V>(this->inner);
  } else {
    return std::nullopt;
  }
}

std::optional<GLWESecretKeyNone> GLWESecretKey::getNone() {
  return get<GLWESecretKeyNone>();
}
std::optional<GLWESecretKeyParameterized> GLWESecretKey::getParameterized() {
  return get<GLWESecretKeyParameterized>();
}
std::optional<GLWESecretKeyNormalized> GLWESecretKey::getNormalized() {
  return get<GLWESecretKeyNormalized>();
}

llvm::hash_code hash_value(const GLWESecretKey &key) {
  return std::visit(overloaded{[](GLWESecretKeyNone sk) {
                                 return llvm::hash_value("GlweSecretKeyNone");
                               },
                               [](GLWESecretKeyParameterized sk) {
                                 return llvm::hash_combine(
                                     "GlweSecretKeyParameterized", sk.dimension,
                                     sk.polySize, sk.identifier);
                               },
                               [](GLWESecretKeyNormalized sk) {
                                 return llvm::hash_combine(
                                     "GlweSecretKeyNormalized", sk.dimension,
                                     sk.polySize, sk.index);
                               }},
                    key.inner);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, GLWESecretKeyNone sk) {
  OS << "sk?";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              GLWESecretKeyParameterized sk) {
  OS << "sk<" << sk.identifier << "," << sk.polySize << "," << sk.dimension
     << ">";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              GLWESecretKeyNormalized sk) {
  OS << "sk[" << sk.index << "]<" << sk.polySize << "," << sk.dimension << ">";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, GLWESecretKey sk) {
  std::visit(overloaded{[&](GLWESecretKeyNone nsk) { OS << nsk; },
                        [&](GLWESecretKeyNormalized nsk) { OS << nsk; },
                        [&](GLWESecretKeyParameterized nsk) { OS << nsk; }},
             sk.inner);
  return OS;
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir

namespace mlir {
AsmPrinter &operator<<(AsmPrinter &p,
                       mlir::concretelang::TFHE::GLWESecretKey key) {
  p.getStream() << key;
  return p;
}

FailureOr<mlir::concretelang::TFHE::GLWESecretKey>
FieldParser<mlir::concretelang::TFHE::GLWESecretKey>::parse(AsmParser &parser) {
  uint64_t dimension = -1, polySize = -1, id = -1;

  if (parser.parseKeyword("sk")) {
    return mlir::failure();
  }

  if (parser.parseOptionalQuestion().succeeded()) { // Parsing none key
    return mlir::concretelang::TFHE::GLWESecretKey::newNone();
  }

  if (parser.parseOptionalLSquare().succeeded()) { // Parsing normalized key
    if (parser.parseInteger(id) || parser.parseRSquare() ||
        parser.parseLess() || parser.parseInteger(polySize) ||
        parser.parseComma() || parser.parseInteger(dimension) ||
        parser.parseGreater()) {
      return mlir::failure();
    } else {
      return mlir::concretelang::TFHE::GLWESecretKey::newNormalized(
          dimension, polySize, id);
    }
  }

  if (parser.parseOptionalLess().succeeded()) { // Parsing parameterized key
    if (parser.parseInteger(id) || parser.parseComma() ||
        parser.parseInteger(polySize) || parser.parseComma() ||
        parser.parseInteger(dimension) || parser.parseGreater()) {
      return mlir::failure();
    } else {
      return mlir::concretelang::TFHE::GLWESecretKey::newParameterized(
          dimension, polySize, id);
    }
  }

  return mlir::failure();
}
} // namespace mlir
