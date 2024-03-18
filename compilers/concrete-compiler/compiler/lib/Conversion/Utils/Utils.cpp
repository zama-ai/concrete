// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Conversion/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"

namespace mlir {
namespace concretelang {

mlir::Type getDynamicMemrefWithUnknownOffset(mlir::RewriterBase &rewriter,
                                             size_t rank) {
  std::vector<int64_t> shape(rank, mlir::ShapedType::kDynamic);
  mlir::AffineExpr expr = rewriter.getAffineSymbolExpr(0);
  for (size_t i = 0; i < rank; i++) {
    expr = expr +
           (rewriter.getAffineDimExpr(i) * rewriter.getAffineSymbolExpr(i + 1));
  }
  return mlir::MemRefType::get(
      shape, rewriter.getI64Type(),
      mlir::AffineMap::get(rank, rank + 1, expr, rewriter.getContext()));
}

// Returns `memref.cast %0 : memref<...xAxT> to memref<...x?xT>`
mlir::Value getCastedMemRef(mlir::RewriterBase &rewriter, mlir::Value value) {
  mlir::Type valueType = value.getType();

  if (auto memrefTy = valueType.dyn_cast_or_null<mlir::MemRefType>()) {
    return rewriter.create<mlir::memref::CastOp>(
        value.getLoc(),
        getDynamicMemrefWithUnknownOffset(rewriter, memrefTy.getShape().size()),
        value);
  } else {
    return value;
  }
}

mlir::Value globalMemrefFromArrayAttr(mlir::RewriterBase &rewriter,
                                      mlir::Location loc,
                                      mlir::ArrayAttr arrAttr) {
  mlir::Type type =
      mlir::RankedTensorType::get({(int)arrAttr.size()}, rewriter.getI64Type());
  std::vector<int64_t> values;
  for (auto a : arrAttr) {
    values.push_back(a.cast<mlir::IntegerAttr>().getValue().getZExtValue());
  }
  auto denseAttr = rewriter.getI64TensorAttr(values);
  auto cstOp = rewriter.create<mlir::arith::ConstantOp>(loc, denseAttr, type);
  auto globalMemref = mlir::bufferization::getGlobalFor(cstOp, 0);
  rewriter.eraseOp(cstOp);
  assert(!mlir::failed(globalMemref));
  auto globalRef = rewriter.create<mlir::memref::GetGlobalOp>(
      loc, (*globalMemref).getType(), (*globalMemref).getName());
  return mlir::concretelang::getCastedMemRef(rewriter, globalRef);
}

} // namespace concretelang
} // namespace mlir
