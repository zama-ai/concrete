// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/RT/IR/RTTypes.h>

namespace mlir {
namespace concretelang {
namespace RT {

void FutureType::print(mlir::AsmPrinter &p) const {
  p << "<";
  p.printType(getElementType());
  p << ">";
}

mlir::Type FutureType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return mlir::Type();

  mlir::Type elementType;

  if (parser.parseType(elementType))
    return mlir::Type();

  if (parser.parseGreater())
    return mlir::Type();

  return get(parser.getContext(), elementType);
}

void PointerType::print(mlir::AsmPrinter &p) const {
  p << "<";
  p.printType(getElementType());
  p << ">";
}

mlir::Type PointerType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return mlir::Type();

  Type elementType;

  if (parser.parseType(elementType))
    return mlir::Type();

  if (parser.parseGreater())
    return mlir::Type();

  return get(parser.getContext(), elementType);
}
} // namespace RT
} // namespace concretelang
} // namespace mlir
