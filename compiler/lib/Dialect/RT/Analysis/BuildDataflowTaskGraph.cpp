// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include <iostream>

#include <mlir/IR/BuiltinOps.h>
#include <zamalang/Dialect/HLFHE/IR/HLFHEDialect.h>
#include <zamalang/Dialect/HLFHE/IR/HLFHEOps.h>
#include <zamalang/Dialect/HLFHE/IR/HLFHETypes.h>
#include <zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h>
#include <zamalang/Dialect/RT/Analysis/Autopar.h>
#include <zamalang/Dialect/RT/IR/RTDialect.h>
#include <zamalang/Dialect/RT/IR/RTOps.h>
#include <zamalang/Dialect/RT/IR/RTTypes.h>
#include <zamalang/Support/Constants.h>
#include <zamalang/Support/math.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/RegionUtils.h>
#include <mlir/Transforms/Utils.h>

#define GEN_PASS_CLASSES
#include <zamalang/Dialect/RT/Analysis/Autopar.h.inc>

namespace mlir {
namespace zamalang {

namespace {

// TODO: adjust these two functions based on cost model
static bool isCandidateForTask(Operation *op) {
  return isa<HLFHE::AddEintIntOp, HLFHE::AddEintOp, HLFHE::SubIntEintOp,
             HLFHE::MulEintIntOp, HLFHE::ApplyLookupTableEintOp,
             HLFHELinalg::MatMulIntEintOp, HLFHELinalg::MatMulEintIntOp,
             HLFHELinalg::AddEintIntOp, HLFHELinalg::AddEintOp,
             HLFHELinalg::SubIntEintOp, HLFHELinalg::NegEintOp,
             HLFHELinalg::MulEintIntOp, HLFHELinalg::ApplyLookupTableEintOp,
             HLFHELinalg::ApplyMultiLookupTableEintOp, HLFHELinalg::Dot>(op);
}

// Identify operations that are beneficial to sink into tasks.  These
// operations must not have side-effects and not be `isCandidateForTask`
static bool isSinkingBeneficiary(Operation *op) {
  return isa<HLFHE::ZeroEintOp, arith::ConstantOp, memref::DimOp, SelectOp,
             mlir::arith::CmpIOp>(op);
}

static bool
extractBeneficiaryOps(Operation *op, SetVector<Value> existingDependencies,
                      SetVector<Operation *> &beneficiaryOps,
                      llvm::SmallPtrSetImpl<Value> &availableValues) {
  if (beneficiaryOps.count(op))
    return true;

  if (!isSinkingBeneficiary(op))
    return false;

  for (Value operand : op->getOperands()) {
    // It is already visible in the kernel, keep going.
    if (availableValues.count(operand))
      continue;
    // Else check whether it can be made available via sinking or already is a
    // dependency.
    Operation *definingOp = operand.getDefiningOp();
    if ((!definingOp ||
         !extractBeneficiaryOps(definingOp, existingDependencies,
                                beneficiaryOps, availableValues)) &&
        !existingDependencies.count(operand))
      return false;
  }
  // We will sink the operation, mark its results as now available.
  beneficiaryOps.insert(op);
  for (Value result : op->getResults())
    availableValues.insert(result);
  return true;
}

LogicalResult sinkOperationsIntoDFTask(RT::DataflowTaskOp taskOp) {
  Region &taskOpBody = taskOp.body();

  // Identify uses from values defined outside of the scope.
  SetVector<Value> sinkCandidates;
  getUsedValuesDefinedAbove(taskOpBody, sinkCandidates);

  SetVector<Operation *> toBeSunk;
  llvm::SmallPtrSet<Value, 4> availableValues;
  for (Value operand : sinkCandidates) {
    Operation *operandOp = operand.getDefiningOp();
    if (!operandOp)
      continue;
    extractBeneficiaryOps(operandOp, sinkCandidates, toBeSunk, availableValues);
  }

  // Insert operations so that the defs get cloned before uses.
  BlockAndValueMapping map;
  OpBuilder builder(taskOpBody);
  for (Operation *op : toBeSunk) {
    OpBuilder::InsertionGuard guard(builder);
    Operation *clonedOp = builder.clone(*op, map);
    for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                 taskOp.body());
    // Once this is sunk, remove all operands of the DFT covered by this
    for (auto result : op->getResults())
      for (auto operand : llvm::enumerate(taskOp.getOperands()))
        if (operand.value() == result) {
          taskOp->eraseOperand(operand.index());
          // Once removed, we assume there are no duplicates
          break;
        }
  }
  return success();
}

// For documentation see Autopar.td
struct BuildDataflowTaskGraphPass
    : public BuildDataflowTaskGraphBase<BuildDataflowTaskGraphPass> {

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](mlir::FuncOp func) {
      if (!func->getAttr("_dfr_work_function_attribute"))
        func.walk(
            [&](mlir::Operation *childOp) { this->processOperation(childOp); });

      // Perform simplifications, in particular DCE here in case some
      // of the operations sunk in tasks are no longer needed in the
      // main function.  If the function fails it only means that
      // nothing was simplified.  Doing this here - rather than later
      // in the compilation pipeline - allows to take advantage of
      // higher level semantics which we can attach to operations
      // (e.g., NoSideEffect on HLFHE::ZeroEintOp).
      IRRewriter rewriter(func->getContext());
      (void)mlir::simplifyRegions(rewriter, func->getRegions());
    });
  }
  BuildDataflowTaskGraphPass(bool debug) : debug(debug){};

protected:
  void processOperation(mlir::Operation *op) {
    if (isCandidateForTask(op)) {
      BlockAndValueMapping map;
      Region &opBody = getOperation().body();
      OpBuilder builder(opBody);

      // Create a DFTask for this operation
      builder.setInsertionPointAfter(op);
      auto dftop = builder.create<RT::DataflowTaskOp>(
          op->getLoc(), op->getResultTypes(), op->getOperands());
      // Add the operation to the task
      OpBuilder tbbuilder(dftop.body());
      Operation *clonedOp = tbbuilder.clone(*op, map);
      // Add sinkable operations to the task
      assert(!failed(sinkOperationsIntoDFTask(dftop)) &&
             "Failing to sink operations into DFT");

      // Add terminator
      tbbuilder.create<RT::DataflowYieldOp>(dftop.getLoc(), mlir::TypeRange(),
                                            op->getResults());
      // Replace the uses of defined values
      for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
        replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                   dftop.body());
      // Replace uses of the values defined by the task
      for (auto pair : llvm::zip(op->getResults(), dftop->getResults()))
        replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                   opBody);
      // Once uses are re-targeted to the task, delete the operation
      op->erase();
    }
  }

  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createBuildDataflowTaskGraphPass(bool debug) {
  return std::make_unique<BuildDataflowTaskGraphPass>(debug);
}

namespace {
// Marker to avoid infinite recursion of the rewriting pattern
static const mlir::StringLiteral kTransformMarker =
    "_internal_RT_FixDataflowTaskOpInputsPattern_marker__";

class FixDataflowTaskOpInputsPattern
    : public mlir::OpRewritePattern<RT::DataflowTaskOp> {
public:
  FixDataflowTaskOpInputsPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<RT::DataflowTaskOp>(
            context, ::mlir::zamalang::DEFAULT_PATTERN_BENEFIT) {}

  LogicalResult
  matchAndRewrite(RT::DataflowTaskOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    if (op->hasAttr(kTransformMarker))
      return failure();

    // Identify which values need to be passed as dependences to the
    // task - this is very conservative and will add constants, index
    // operations, etc.  A simplification will occur later.
    SetVector<Value> deps;
    getUsedValuesDefinedAbove(op.body(), deps);
    auto newop = rewriter.create<RT::DataflowTaskOp>(
        op.getLoc(), op.getResultTypes(), deps.getArrayRef());
    rewriter.mergeBlocks(op.getBody(), newop.getBody(),
                         newop.getBody()->getArguments());
    rewriter.replaceOp(op, {newop.getResults()});

    // Mark this as processed to prevent infinite loop
    newop.getOperation()->setAttr(kTransformMarker, rewriter.getUnitAttr());
    return success();
  }
};
} // namespace

namespace {
// For documentation see Autopar.td
struct FixupDataflowTaskOpsPass
    : public FixupDataflowTaskOpsBase<FixupDataflowTaskOpsPass> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<FixDataflowTaskOpInputsPattern>(context);

    if (mlir::applyPatternsAndFoldGreedily(module, std::move(patterns))
            .failed())
      signalPassFailure();

    // Clear mark and sink any newly created constants or indexing
    // operations, etc. to reduce the number of input dependences to
    // the task
    module->walk([](RT::DataflowTaskOp op) {
      op.getOperation()->removeAttr(kTransformMarker);
      assert(!failed(sinkOperationsIntoDFTask(op)) &&
             "Failing to sink operations into DFT");
    });
  }

  FixupDataflowTaskOpsPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createFixupDataflowTaskOpsPass(bool debug) {
  return std::make_unique<FixupDataflowTaskOpsPass>(debug);
}

} // end namespace zamalang
} // end namespace mlir
