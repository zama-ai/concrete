// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/TFHE/IR/TFHETypes.h>

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
  auto crt = getCrtDecomposition();
  if (!crt.empty()) {
    p << "crt=[";
    for (auto c : crt.drop_back(1)) {
      printSigned(p, c);
      p << ",";
    }
    printSigned(p, crt.back());
    p << "]";
  }
  p << "{";
  printSigned(p, getDimension());
  p << ",";
  printSigned(p, getPolynomialSize());
  p << ",";
  printSigned(p, getBits());
  p << "}";

  p << "{";
  printSigned(p, getP());
  p << "}>";
}

mlir::Type GLWECipherTextType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return mlir::Type();

  // Parse for the crt decomposition if any
  std::vector<int64_t> crtDecomposition;
  if (!parser.parseOptionalKeyword("crt")) {
    if (parser.parseEqual() || parser.parseLSquare())
      return mlir::Type();
    while (true) {
      signed c = -1;
      if (parser.parseOptionalKeyword("_") && parser.parseInteger(c)) {
        return mlir::Type();
      }
      crtDecomposition.push_back(c);
      if (parser.parseOptionalComma()) {
        if (parser.parseRSquare()) {
          return mlir::Type();
        } else {
          break;
        }
      }
    }
  }

  if (parser.parseLBrace())
    return mlir::Type();

  // First parameters block
  int dimension = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(dimension))
    return mlir::Type();
  if (parser.parseComma())
    return mlir::Type();
  int polynomialSize = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(polynomialSize))
    return mlir::Type();
  if (parser.parseComma())
    return mlir::Type();
  int bits = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(bits))
    return mlir::Type();
  if (parser.parseRBrace())
    return mlir::Type();

  // Next parameters block
  if (parser.parseLBrace())
    return mlir::Type();
  int p = -1;
  if (parser.parseInteger(p))
    return mlir::Type();
  if (parser.parseRBrace())
    return mlir::Type();

  if (parser.parseGreater())
    return mlir::Type();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  return getChecked(loc, loc.getContext(), dimension, polynomialSize, bits, p,
                    llvm::ArrayRef<int64_t>(crtDecomposition));
}
} // namespace TFHE
} // namespace concretelang
} // namespace mlir
