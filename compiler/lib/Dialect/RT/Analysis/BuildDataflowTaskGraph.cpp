// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h>
#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>
#include <concretelang/Support/Constants.h>
#include <concretelang/Support/math.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/RegionUtils.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/RT/Analysis/Autopar.h.inc>

namespace mlir {
namespace concretelang {

namespace {

// TODO: adjust these two functions based on cost model
static bool isCandidateForTask(Operation *op) {
  return isa<
      FHE::ApplyLookupTableEintOp, FHELinalg::MatMulEintIntOp,
      FHELinalg::AddEintIntOp, FHELinalg::AddEintOp, FHELinalg::SubIntEintOp,
      FHELinalg::SubEintIntOp, FHELinalg::SubEintOp, FHELinalg::NegEintOp,
      FHELinalg::MulEintIntOp, FHELinalg::ApplyLookupTableEintOp,
      FHELinalg::ApplyMultiLookupTableEintOp,
      FHELinalg::ApplyMappedLookupTableEintOp, FHELinalg::Dot,
      FHELinalg::MatMulEintIntOp, FHELinalg::MatMulIntEintOp, FHELinalg::SumOp,
      FHELinalg::ConcatOp, FHELinalg::Conv2dOp, FHELinalg::TransposeOp>(op);
}

/// Identify operations that are beneficial to aggregate into tasks.  These
/// operations must not have side-effects and not be `isCandidateForTask`
static bool isAggregatingBeneficiary(Operation *op) {
  return isa<FHE::ZeroEintOp, FHE::ZeroTensorOp, FHE::AddEintIntOp,
             FHE::AddEintOp, FHE::SubIntEintOp, FHE::SubEintIntOp,
             FHE::MulEintIntOp, FHE::SubEintOp, FHE::NegEintOp,
             FHELinalg::FromElementOp, arith::ConstantOp, memref::DimOp,
             arith::SelectOp, mlir::arith::CmpIOp, memref::GetGlobalOp,
             memref::CastOp>(op);
}

static bool
aggregateBeneficiaryOps(Operation *op, SetVector<Operation *> &beneficiaryOps,
                        llvm::SmallPtrSetImpl<Value> &availableValues) {
  if (beneficiaryOps.count(op))
    return true;

  if (!isAggregatingBeneficiary(op))
    return false;

  // Gather the new potential dependences created by sinking this op.
  llvm::SmallPtrSet<Value, 4> newDependencesIfSunk;
  for (Value operand : op->getOperands())
    if (!availableValues.count(operand))
      newDependencesIfSunk.insert(operand);

  // We further attempt to sink any new dependence
  for (auto dep : newDependencesIfSunk) {
    Operation *definingOp = dep.getDefiningOp();
    if (definingOp)
      aggregateBeneficiaryOps(definingOp, beneficiaryOps, availableValues);
  }

  // We will sink the operation, mark its results as now available.
  beneficiaryOps.insert(op);
  for (Value result : op->getResults())
    availableValues.insert(result);
  return true;
}

static bool isFunctionCallName(OpOperand *use, StringRef name) {
  func::CallOp call = dyn_cast_or_null<mlir::func::CallOp>(use->getOwner());
  if (!call)
    return false;
  SymbolRefAttr sym = call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return false;
  func::FuncOp called = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(call, sym));
  if (!called)
    return false;
  return called.getName() == name;
}

static void getAliasedUses(Value val, DenseSet<OpOperand *> &aliasedUses) {
  for (auto &use : val.getUses()) {
    aliasedUses.insert(&use);
    if (isa<memref::CastOp, memref::ViewOp, memref::SubViewOp>(use.getOwner()))
      getAliasedUses(use.getOwner()->getResult(0), aliasedUses);
  }
}

static bool aggregateOutputMemrefAllocations(
    Operation *op, SetVector<Operation *> &beneficiaryOps,
    llvm::SmallPtrSetImpl<Value> &availableValues, RT::DataflowTaskOp taskOp) {
  if (beneficiaryOps.count(op))
    return true;

  if (!isa<mlir::memref::AllocOp>(op))
    return false;

  Value val = op->getResults().front();
  DenseSet<OpOperand *> aliasedUses;
  getAliasedUses(val, aliasedUses);

  // Helper function checking if a memref use writes to memory
  auto hasMemoryWriteEffect = [&](OpOperand *use) {
    // Call ops targeting concrete-ffi do not have memory effects
    // interface, so handle apart.
    // TODO: this could be handled better in BConcrete or higher.
    if (isFunctionCallName(use, "memref_expand_lut_in_trivial_glwe_ct_u64") ||
        isFunctionCallName(use, "memref_add_lwe_ciphertexts_u64") ||
        isFunctionCallName(use, "memref_add_plaintext_lwe_ciphertext_u64") ||
        isFunctionCallName(use, "memref_mul_cleartext_lwe_ciphertext_u64") ||
        isFunctionCallName(use, "memref_negate_lwe_ciphertext_u64") ||
        isFunctionCallName(use, "memref_keyswitch_lwe_u64") ||
        isFunctionCallName(use, "memref_bootstrap_lwe_u64"))
      if (use->getOwner()->getOperand(0) == use->get())
        return true;

    if (isFunctionCallName(use, "memref_copy_one_rank"))
      if (use->getOwner()->getOperand(1) == use->get())
        return true;

    // Otherwise we rely on the memory effect interface
    auto effectInterface = dyn_cast<MemoryEffectOpInterface>(use->getOwner());
    if (!effectInterface)
      return false;
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    effectInterface.getEffects(effects);
    for (auto eff : effects)
      if (isa<MemoryEffects::Write>(eff.getEffect()) &&
          eff.getValue() == use->get())
        return true;
    return false;
  };

  // We need to check if this allocated memref is written in this task.
  // TODO: for now we'll assume that we don't do partial writes or read/writes.
  for (auto use : aliasedUses)
    if (hasMemoryWriteEffect(use) &&
        use->getOwner()->getParentOfType<RT::DataflowTaskOp>() == taskOp) {
      // We will sink the operation, mark its results as now available.
      beneficiaryOps.insert(op);
      for (Value result : op->getResults())
        availableValues.insert(result);
      return true;
    }
  return false;
}

LogicalResult coarsenDFTask(RT::DataflowTaskOp taskOp) {
  Region &taskOpBody = taskOp.body();

  // Identify uses from values defined outside of the scope.
  SetVector<Value> sinkCandidates;
  getUsedValuesDefinedAbove(taskOpBody, sinkCandidates);

  SetVector<Operation *> toBeSunk;
  llvm::SmallPtrSet<Value, 4> availableValues(sinkCandidates.begin(),
                                              sinkCandidates.end());
  for (Value operand : sinkCandidates) {
    Operation *operandOp = operand.getDefiningOp();
    if (!operandOp)
      continue;
    aggregateBeneficiaryOps(operandOp, toBeSunk, availableValues);
    aggregateOutputMemrefAllocations(operandOp, toBeSunk, availableValues,
                                     taskOp);
  }

  // Insert operations so that the defs get cloned before uses.
  BlockAndValueMapping map;
  OpBuilder builder(taskOpBody);
  for (Operation *op : toBeSunk) {
    OpBuilder::InsertionGuard guard(builder);
    Operation *clonedOp = builder.clone(*op, map);
    for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                 taskOpBody);
  }

  SetVector<Value> deps;
  getUsedValuesDefinedAbove(taskOpBody, deps);
  taskOp->setOperands(deps.takeVector());

  return success();
}

/// For documentation see Autopar.td
struct BuildDataflowTaskGraphPass
    : public BuildDataflowTaskGraphBase<BuildDataflowTaskGraphPass> {

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](mlir::func::FuncOp func) {
      if (!func->getAttr("_dfr_work_function_attribute"))
        func.walk(
            [&](mlir::Operation *childOp) { this->processOperation(childOp); });

      // Perform simplifications, in particular DCE here in case some
      // of the operations sunk in tasks are no longer needed in the
      // main function.  If the function fails it only means that
      // nothing was simplified.  Doing this here - rather than later
      // in the compilation pipeline - allows to take advantage of
      // higher level semantics which we can attach to operations
      // (e.g., NoSideEffect on FHE::ZeroEintOp).
      IRRewriter rewriter(func->getContext());
      (void)mlir::simplifyRegions(rewriter, func->getRegions());
    });
  }
  BuildDataflowTaskGraphPass(bool debug) : debug(debug){};

protected:
  void processOperation(mlir::Operation *op) {
    if (isCandidateForTask(op)) {
      BlockAndValueMapping map;
      Region &opBody = getOperation().getBody();
      OpBuilder builder(opBody);

      // Create a DFTask for this operation
      builder.setInsertionPointAfter(op);
      auto dftop = builder.create<RT::DataflowTaskOp>(
          op->getLoc(), op->getResultTypes(), op->getOperands());

      // Add the operation to the task
      OpBuilder tbbuilder(dftop.body());
      Operation *clonedOp = tbbuilder.clone(*op, map);

      // Coarsen granularity by aggregating all dependence related
      // lower-weight operations.
      assert(!failed(coarsenDFTask(dftop)) &&
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
/// For documentation see Autopar.td
struct FixupDataflowTaskOpsPass
    : public FixupDataflowTaskOpsBase<FixupDataflowTaskOpsPass> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    module->walk([](RT::DataflowTaskOp op) {
      assert(!failed(coarsenDFTask(op)) &&
             "Failing to sink operations into DFT");
    });

    // Finally clear up any remaining alloc/dealloc ops that are
    // meaningless
    SetVector<Operation *> eraseOps;
    module->walk([&](memref::AllocOp op) {
      // If this memref.alloc's only use left is the
      // dealloc, erase both.
      if (op->hasOneUse() &&
          isa<mlir::memref::DeallocOp>(op->use_begin()->getOwner())) {
        eraseOps.insert(op->use_begin()->getOwner());
        eraseOps.insert(op);
      }
    });
    for (auto op : eraseOps)
      op->erase();
  }

  FixupDataflowTaskOpsPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createFixupDataflowTaskOpsPass(bool debug) {
  return std::make_unique<FixupDataflowTaskOpsPass>(debug);
}

} // end namespace concretelang
} // end namespace mlir
