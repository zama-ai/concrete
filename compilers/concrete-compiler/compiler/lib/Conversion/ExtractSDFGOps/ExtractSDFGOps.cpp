// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"
#include "concretelang/Dialect/SDFG/Interfaces/SDFGConvertibleInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <concretelang/Dialect/Concrete/IR/ConcreteDialect.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteOps.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteTypes.h>

namespace SDFG = mlir::concretelang::SDFG;
namespace Concrete = mlir::concretelang::Concrete;

namespace {
enum class StreamMappingKind { ON_DEVICE, TO_DEVICE, SPLICE, TO_HOST, NONE };

SDFG::MakeStream makeStream(mlir::ImplicitLocOpBuilder &builder,
                            SDFG::StreamKind kind, mlir::Type type,
                            mlir::Value dfg, unsigned &streamNumber) {
  SDFG::StreamType streamType = builder.getType<SDFG::StreamType>(type);
  mlir::StringAttr name = builder.getStringAttr(llvm::Twine("stream") +
                                                llvm::Twine(streamNumber++));

  return builder.create<SDFG::MakeStream>(streamType, dfg, name, kind);
}

/// Unrolls entirely all scf loops, which directly contain an
/// SDFG-convertible operation and whose bounds are static.
void unrollLoopsWithSDFGConvertibleOps(mlir::func::FuncOp func) {
  mlir::DenseSet<mlir::scf::ForOp> unrollCandidates;

  // Identify loops with SDFG-convertible ops
  func.walk([&](SDFG::SDFGConvertibleOpInterface convertible) {
    for (mlir::Operation *parent = convertible->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (mlir::scf::ForOp forOp = llvm::dyn_cast<mlir::scf::ForOp>(parent)) {
        unrollCandidates.insert(forOp);
      }
    }
  });

  // Fully unroll all the loops if its bounds are static
  for (mlir::scf::ForOp forOp : unrollCandidates) {
    mlir::arith::ConstantIndexOp lb =
        forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    mlir::arith::ConstantIndexOp ub =
        forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    mlir::arith::ConstantIndexOp step =
        forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();

    if (!lb || !ub || !step)
      continue;

    int64_t ilb = lb.value();
    int64_t iub = ub.value();
    int64_t istep = step.value();

    // Unrolling requires positive bounds and step
    if (ilb < 0 || iub < 0 || istep <= 0)
      continue;

    int64_t unrollFactor = ((iub - ilb) + (istep - 1)) / istep;

    if (unrollFactor == 0)
      continue;

    if (mlir::loopUnrollByFactor(forOp, (uint64_t)unrollFactor).failed())
      continue;
  }
}

StreamMappingKind determineStreamMappingKind(mlir::Value v) {
  // Determine stream type for operands:
  //
  //  - If an operand is produced by a non-convertible op, there
  //    needs to be just a host-to-device stream
  //
  //  - If an operand is produced by a convertible op and there
  //    are no other consumers, a device-to-device stream may be
  //    used
  //
  //  - If an operand is produced by a convertible op and there is
  //    at least one other non-convertible consumer, there needs
  //    to be a device-to-host stream, and a host-to-device stream
  //
  if (llvm::dyn_cast_or_null<SDFG::SDFGConvertibleOpInterface>(
          v.getDefiningOp())) {
    // All convertible consumers?
    if (llvm::all_of(v.getUses(), [](mlir::OpOperand &o) {
          return !!llvm::dyn_cast_or_null<SDFG::SDFGConvertibleOpInterface>(
              o.getOwner());
        })) {
      return StreamMappingKind::ON_DEVICE;
    }
    // All non-convertible consumers?
    else if (llvm::all_of(v.getUses(), [](mlir::OpOperand &o) {
               return !llvm::dyn_cast_or_null<SDFG::SDFGConvertibleOpInterface>(
                   o.getOwner());
             })) {
      return StreamMappingKind::TO_HOST;
    }
    // Mix of convertible and non-convertible users
    else {
      return StreamMappingKind::SPLICE;
    }
  } else {
    if (llvm::any_of(v.getUses(), [](mlir::OpOperand &o) {
          return !!llvm::dyn_cast_or_null<SDFG::SDFGConvertibleOpInterface>(
              o.getOwner());
        })) {
      return StreamMappingKind::TO_DEVICE;
    } else {
      return StreamMappingKind::NONE;
    }
  }
}

void setInsertionPointAfterValueOrRestore(mlir::OpBuilder &builder,
                                          mlir::Value v,
                                          mlir::OpBuilder::InsertPoint &pos) {
  if (v.isa<mlir::BlockArgument>())
    builder.restoreInsertionPoint(pos);
  else
    builder.setInsertionPointAfterValue(v);
}

struct ExtractSDFGOpsPass : public ExtractSDFGOpsBase<ExtractSDFGOpsPass> {
  bool unroll;

  ExtractSDFGOpsPass(bool unroll) : unroll(unroll) {}

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func.getContext());

    if (unroll)
      unrollLoopsWithSDFGConvertibleOps(func);

    mlir::DenseMap<mlir::Value, SDFG::MakeStream> processOutMapping;
    mlir::DenseMap<mlir::Value, SDFG::MakeStream> processInMapping;
    mlir::DenseMap<mlir::Value, mlir::Value> replacementMapping;

    llvm::SmallVector<SDFG::SDFGConvertibleOpInterface> convertibleOps;

    unsigned streamNumber = 0;

    // Restrict SDFG conversion to cases where the SDFG graph includes
    // operations with sufficient computational complexity to benefit
    // from offloading to an accelerator.
    auto isOpInBootstrappingSDFG = [&](mlir::Operation *op) -> bool {
      mlir::scf::ForOp loopParent =
          llvm::dyn_cast_or_null<mlir::scf::ForOp>(op->getParentOp());
      if (loopParent) {
        for (mlir::Operation &bop : loopParent.getBody()->getOperations())
          if (llvm::isa<Concrete::BatchedBootstrapLweTensorOp,
                        Concrete::BatchedMappedBootstrapLweTensorOp,
                        Concrete::BatchedKeySwitchLweTensorOp>(bop))
            return true;
        return false;
      } else {
        return true;
      }
    };
    // If we will generate SDFG ops for a loop, then the loop and
    // all enclosing loops must not be parallelized as SDFG access
    // ops (put/get) need to be issued in serial order.
    auto restrictParallelLoopsWithSDFGConvertibleOps =
        [&](mlir::Operation *op) {
          mlir::scf::ForOp loopParent =
              llvm::dyn_cast_or_null<mlir::scf::ForOp>(op->getParentOp());
          if (loopParent) {
            loopParent->setAttr("parallel", rewriter.getBoolAttr(false));
            for (mlir::Operation *parent = loopParent->getParentOp(); parent;
                 parent = parent->getParentOp())
              if (mlir::scf::ForOp forOp =
                      llvm::dyn_cast<mlir::scf::ForOp>(parent))
                forOp->setAttr("parallel", rewriter.getBoolAttr(false));
          }
        };
    func.walk([&](SDFG::SDFGConvertibleOpInterface op) {
      if (isOpInBootstrappingSDFG(op)) {
        restrictParallelLoopsWithSDFGConvertibleOps(op);
        convertibleOps.push_back(op);
      }
    });

    if (convertibleOps.size() == 0)
      return;

    // Insert Prelude
    rewriter.setInsertionPointToStart(&func.getBlocks().front());
    mlir::Value dfg = rewriter.create<SDFG::Init>(
        func.getLoc(), rewriter.getType<SDFG::DFGType>());
    SDFG::Start start = rewriter.create<SDFG::Start>(func.getLoc(), dfg);

    rewriter.setInsertionPoint(func.getBlocks().back().getTerminator());
    rewriter.create<SDFG::Shutdown>(func.getLoc(), dfg);

    mlir::ImplicitLocOpBuilder ilb(func.getLoc(), rewriter);

    auto mapValueToStreams = [&](mlir::Value v) {
      if (processInMapping.find(v) != processInMapping.end() ||
          processOutMapping.find(v) != processOutMapping.end())
        return;

      StreamMappingKind smk = determineStreamMappingKind(v);

      SDFG::MakeStream prodOutStream;
      SDFG::MakeStream consInStream;

      mlir::OpBuilder::InsertPoint getPos;
      if (smk == StreamMappingKind::SPLICE ||
          smk == StreamMappingKind::TO_HOST) {
        ilb.setInsertionPoint(start);
        prodOutStream = makeStream(ilb,
                                   (smk == StreamMappingKind::TO_HOST)
                                       ? SDFG::StreamKind::device_to_host
                                       : SDFG::StreamKind::device_to_both,
                                   v.getType(), dfg, streamNumber);
        processOutMapping.insert({v, prodOutStream});

        ilb.setInsertionPointAfter(start);
        mlir::OpBuilder::InsertPoint pos = ilb.saveInsertionPoint();
        setInsertionPointAfterValueOrRestore(ilb, v, pos);
        mlir::Value newOutVal =
            ilb.create<SDFG::Get>(v.getType(), prodOutStream.getResult());
        replacementMapping.insert({v, newOutVal});
        getPos = ilb.saveInsertionPoint();
      } else if (smk == StreamMappingKind::ON_DEVICE) {
        ilb.setInsertionPoint(start);
        prodOutStream = makeStream(ilb, SDFG::StreamKind::on_device,
                                   v.getType(), dfg, streamNumber);
        processOutMapping.insert({v, prodOutStream});
      }

      if (smk == StreamMappingKind::ON_DEVICE ||
          smk == StreamMappingKind::SPLICE) {
        processInMapping.insert({v, prodOutStream});
      } else if (smk == StreamMappingKind::TO_DEVICE) {
        ilb.setInsertionPoint(start);
        consInStream = makeStream(ilb, SDFG::StreamKind::host_to_device,
                                  v.getType(), dfg, streamNumber);
        processInMapping.insert({v, consInStream});

        ilb.setInsertionPointAfter(start);
        mlir::OpBuilder::InsertPoint pos = ilb.saveInsertionPoint();
        setInsertionPointAfterValueOrRestore(ilb, v, pos);
        ilb.create<SDFG::Put>(consInStream.getResult(), v);
      }
    };

    for (SDFG::SDFGConvertibleOpInterface convertibleOp : convertibleOps) {
      llvm::SmallVector<mlir::Value> ins;
      llvm::SmallVector<mlir::Value> outs;
      ilb.setLoc(convertibleOp.getLoc());

      for (mlir::Value res : convertibleOp->getResults()) {
        mapValueToStreams(res);
        outs.push_back(processOutMapping.find(res)->second.getResult());
      }

      for (mlir::Value operand : convertibleOp->getOperands()) {
        mapValueToStreams(operand);
        ins.push_back(processInMapping.find(operand)->second.getResult());
      }

      ilb.setInsertionPoint(start);
      SDFG::MakeProcess process = convertibleOp.convert(ilb, dfg, ins, outs);

      assert(process && "Conversion to SDFG operation failed");
    }

    for (auto it : replacementMapping) {
      it.first.replaceAllUsesWith(it.second);
    }

    (void)mlir::simplifyRegions(rewriter, func->getRegions());

      func.walk([&](mlir::scf::ForOp loop) {
      for (mlir::Operation &op : loop.getBody()->getOperations()) {
	if (llvm::isa<Concrete::EncodeExpandLutForBootstrapTensorOp>(op)) {
	  loop->setAttr("parallel", rewriter.getBoolAttr(true));
	  return;
	}
      }
    });
  }
};
} // namespace

namespace mlir {
namespace concretelang {

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createExtractSDFGOpsPass(bool unroll) {
  return std::make_unique<ExtractSDFGOpsPass>(unroll);
}
} // namespace concretelang
} // namespace mlir
