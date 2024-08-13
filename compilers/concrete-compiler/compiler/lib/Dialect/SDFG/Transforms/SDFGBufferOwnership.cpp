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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"
#include "concretelang/Dialect/SDFG/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Support/CompilerEngine.h"
#include <concretelang/Dialect/SDFG/Transforms/Passes.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace SDFG = mlir::concretelang::SDFG;

namespace mlir {
namespace concretelang {
namespace {

static void getAliasedUses(Value val, DenseSet<OpOperand *> &aliasedUses) {
  for (auto &use : val.getUses()) {
    aliasedUses.insert(&use);
    if (dyn_cast<ViewLikeOpInterface>(use.getOwner()))
      getAliasedUses(use.getOwner()->getResult(0), aliasedUses);
  }
}

static func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

struct SDFGBufferOwnershipPass
    : public SDFGBufferOwnershipBase<SDFGBufferOwnershipPass> {

  void runOnOperation() override {
    auto module = getOperation();
    std::vector<Operation *> deallocOps;

    // Find all SDFG put operations that use a buffer only used for
    // this operation, then deallocated. In such cases there is no
    // need to copy the data again in the runtime and we can take
    // ownership of the buffer instead, removing the deallocation and
    // allowing the runtime to deallocate when appropriate.
    module.walk([&](mlir::memref::DeallocOp op) {
      DominanceInfo domInfo(op);
      Value alloc = op.getOperand();
      DenseSet<OpOperand *> aliasedUses;
      getAliasedUses(alloc, aliasedUses);

      // Check if this memref is used in a SDFG put operation
      for (auto use : aliasedUses) {
        if (isa<mlir::func::CallOp>(use->getOwner())) {
          mlir::func::CallOp callOp = cast<func::CallOp>(use->getOwner());
          mlir::func::FuncOp funcOp = getCalledFunction(callOp);
          std::string putName = "stream_emulator_put_memref";
          if (funcOp.getName().str().compare(0, putName.size(), putName) == 0) {
            // If the put operation dominates the deallocation, then
            // ownership of the data can be transferred to the runtime
            // and deallocation can be removed. We mark the ownership
            // flag in the PUT operation to notify the runtime that it
            // gets ownership.
            if (domInfo.properlyDominates(callOp, op)) {
              deallocOps.push_back(op);
              OpBuilder builder(callOp);
              mlir::Value cst1 = builder.create<mlir::arith::ConstantOp>(
                  callOp.getLoc(), builder.getI64IntegerAttr(1));
              callOp->setOperand(2, cst1);
            }
            return;
          }
        }
      }
    });

    for (auto dop : deallocOps) {
      dop->erase();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createSDFGBufferOwnershipPass() {
  return std::make_unique<SDFGBufferOwnershipPass>();
}

} // end namespace concretelang
} // end namespace mlir
