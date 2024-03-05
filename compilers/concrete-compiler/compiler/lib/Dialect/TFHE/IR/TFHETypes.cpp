// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/TFHE/IR/TFHETypes.h>
#include <mlir/IR/DialectImplementation.h>

namespace mlir {
namespace concretelang {
namespace TFHE {

void printSigned(mlir::AsmPrinter &p, signed i) {
  if (i == -1)
    p << "_";
  else
    p << i;
}

void GLWECipherTextType::print(mlir::AsmPrinter &p) const {
  p << "<";
  p << getKey();
  p << ">";
}

mlir::Type GLWECipherTextType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return mlir::Type();

  // First parameters block
  FailureOr<mlir::concretelang::TFHE::GLWESecretKey> maybeKey =
      FieldParser<GLWESecretKey>::parse(parser);
  if (failed(maybeKey))
    return mlir::Type();

  if (parser.parseGreater())
    return mlir::Type();

  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  return getChecked(loc, loc.getContext(), maybeKey.value());
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir
