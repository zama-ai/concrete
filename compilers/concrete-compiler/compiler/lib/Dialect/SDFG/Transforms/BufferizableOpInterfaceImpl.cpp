// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"
#include "concretelang/Dialect/SDFG/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Support/CompilerEngine.h"
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace SDFG = mlir::concretelang::SDFG;

namespace mlir {
namespace concretelang {
namespace SDFG {
namespace {} // namespace
} // namespace SDFG
} // namespace concretelang
} // namespace mlir

namespace {
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
mlir::Value getCastedMemRef(mlir::RewriterBase &rewriter, mlir::Location loc,
                            mlir::Value value) {
  mlir::Type valueType = value.getType();
  if (auto memrefTy = valueType.dyn_cast_or_null<mlir::MemRefType>()) {
    return rewriter.create<mlir::memref::CastOp>(
        loc,
        getDynamicMemrefWithUnknownOffset(rewriter, memrefTy.getShape().size()),
        value);
  } else {
    return value;
  }
}

char stream_emulator_get_memref[] = "stream_emulator_get_memref";
char stream_emulator_get_memref_batch[] = "stream_emulator_get_memref_batch";

template <typename Op, char const *funcName, char const *funcName_batch>
struct BufferizableWithCallOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizableWithCallOpInterface<Op, funcName, funcName_batch>, Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Unknown;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto loc = op->getLoc();

    // TODO: For now we allocate for the result of GET but we might be
    // able to avoid the copy depending on the stream semantics.
    auto resTensorType =
        op->getResultTypes()[0].template cast<mlir::TensorType>();
    char const *fname;
    if (resTensorType.getRank() == 1)
      fname = funcName;
    else if (resTensorType.getRank() == 2)
      fname = funcName_batch;
    else
      return mlir::failure();
    auto outMemrefType = MemRefType::get(resTensorType.getShape(),
                                         resTensorType.getElementType());
    auto outMemref = options.createAlloc(rewriter, loc, outMemrefType, {});
    if (mlir::failed(outMemref)) {
      return mlir::failure();
    }

    // The last operand is the result
    mlir::SmallVector<mlir::Value> operands(op->getOperands());
    operands.push_back(getCastedMemRef(rewriter, loc, *outMemref));

    mlir::FunctionType funcType = mlir::FunctionType::get(
        rewriter.getContext(), mlir::ValueRange{operands}.getTypes(),
        mlir::TypeRange());
    if (insertForwardDeclaration(op, rewriter, fname, funcType).failed())
      return ::mlir::failure();
    rewriter.create<mlir::func::CallOp>(loc, fname, mlir::TypeRange{},
                                        operands);
    replaceOpWithBufferizedValues(rewriter, op, *outMemref);

    return success();
  }
};

} // namespace

void mlir::concretelang::SDFG::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, SDFG::SDFGDialect *dialect) {
    SDFG::Get::attachInterface<
        BufferizableWithCallOpInterface<SDFG::Get, stream_emulator_get_memref,
                                        stream_emulator_get_memref_batch>>(
        *ctx);
  });
}
