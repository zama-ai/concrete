// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.h"
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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
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
  // if it's a linalg.genric operation with encrypted inputs
  if (auto genericOp = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
    for (auto input : genericOp.getInputs()) {
      if ((input.getType().isa<ShapedType>() &&
           mlir::dyn_cast<ShapedType>(input.getType())
               .getElementType()
               .isa<FHE::FheIntegerInterface>()) ||
          input.getType().isa<FHE::FheIntegerInterface>()) {
        return true;
      }
    }
  }
  return isa<FHE::ApplyLookupTableEintOp>(op);
}

/// Identify operations that are beneficial to aggregate into tasks.  These
/// operations must not have side-effects and not be `isCandidateForTask`
static bool isAggregatingBeneficiary(Operation *op) {
  return isa<FHE::ZeroEintOp, FHE::ZeroTensorOp, FHE::AddEintIntOp,
             FHE::AddEintOp, FHE::SubIntEintOp, FHE::SubEintIntOp,
             FHE::MulEintIntOp, FHE::SubEintOp, FHE::NegEintOp,
             FHELinalg::FromElementOp, arith::ConstantOp, arith::SelectOp,
             mlir::arith::CmpIOp>(op);
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

LogicalResult coarsenDFTask(RT::DataflowTaskOp taskOp) {
  Region &taskOpBody = taskOp.getBody();

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
  }

  // Insert operations so that the defs get cloned before uses.
  IRMapping map;
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
        func.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *childOp) {
          return this->processOperation(childOp);
        });

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

protected:
  mlir::WalkResult processOperation(mlir::Operation *op) {
    if (isCandidateForTask(op)) {
      IRMapping map;
      Region &opBody = getOperation().getBody();
      OpBuilder builder(opBody);

      // Create a DFTask for this operation
      builder.setInsertionPointAfter(op);
      auto dftop = builder.create<RT::DataflowTaskOp>(
          op->getLoc(), op->getResultTypes(), op->getOperands());

      // Add the operation to the task
      OpBuilder tbbuilder(dftop.getBody());
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
                                   dftop.getBody());
      // Replace uses of the values defined by the task
      for (auto pair : llvm::zip(op->getResults(), dftop->getResults()))
        replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                   opBody);
      // Once uses are re-targeted to the task, delete the operation
      op->erase();
    }
    return mlir::WalkResult::advance();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createBuildDataflowTaskGraphPass() {
  return std::make_unique<BuildDataflowTaskGraphPass>();
}

} // end namespace concretelang
} // end namespace mlir
