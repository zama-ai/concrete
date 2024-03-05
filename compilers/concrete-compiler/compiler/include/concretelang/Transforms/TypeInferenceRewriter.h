// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

// Infrastructure for rewriting IR after running type inference using
// `ForwardTypeInferenceAnalysis` and `BackwardTypeInferenceAnalysis`
//
// The `TypeInferenceRewriter` class can be used as a base for a
// rewriter applying the types inferred through type inference. Final
// types are inferred by retrieving the inferred types for the
// respective values from the data flow solver and by invoking the type
// resolver one last time.
//
// Conflicts can be handled by overriding the `handleConflict`
// method. By default, this method generates a
// `TypeInference.unresolved_conflict` operation.

#ifndef CONCRETELANG_TRANSFORMS_TYPEINFERENCEREWRITER_H
#define CONCRETELANG_TRANSFORMS_TYPEINFERENCEREWRITER_H

#include <concretelang/Analysis/TypeInferenceAnalysis.h>
#include <concretelang/Dialect/TypeInference/IR/TypeInferenceOps.h>

#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
namespace concretelang {

// Rewrites the types in a module based on the results contained in
// the states of a `DataFlow` solver that has run type analysis.
class TypeInferenceRewriter {
public:
  TypeInferenceRewriter(const mlir::DataFlowSolver &solver,
                        TypeResolver &typeResolver)
      : solver(solver), typeResolver(typeResolver) {}

  virtual ~TypeInferenceRewriter() {}

  // Rewrites a module. If rewriting fails, e.g., due to unresolved
  // types, an error is emitted for the corresponding op, the module
  // is left partially rewritten and the function returns
  // `mlir::failure()`.
  mlir::LogicalResult rewrite(mlir::ModuleOp module) {
    mlir::IRRewriter rewriter(module.getContext());

    if (module
            .walk([&](mlir::func::FuncOp func) {
              if (rewrite(func, rewriter).failed()) {
                return mlir::WalkResult::interrupt();
              } else {
                return mlir::WalkResult::advance();
              }
            })
            .wasInterrupted()) {
      return mlir::failure();
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    return this->postRewriteHook(rewriter, module, module);
  }

protected:
  // Rewrite a single function
  mlir::LogicalResult rewrite(mlir::func::FuncOp func,
                              mlir::IRRewriter &rewriter) {
    rewriter.setInsertionPointAfter(func);

    // Gather inferred types an run the solver a last time on the
    // function to get the actual types for the arguments and return
    // values
    LocalInferenceState inferredTypes =
        TypeInferenceUtils::getLocalInferenceState(solver, func);
    LocalInferenceState resolvedTypes =
        typeResolver.resolve(func, inferredTypes);

    llvm::SmallVector<mlir::Type> argTypes =
        resolvedTypes.lookup(func.getBody().getArguments());

    llvm::SmallVector<mlir::Type> returnTypes = resolvedTypes.lookup(
        func.getRegion().getBlocks().front().getTerminator()->getOperands());

    if (checkAllResolved(func, argTypes, "argument").failed() ||
        checkAllResolved(func, returnTypes, "return value").failed()) {
      return mlir::failure();
    }

    // Temporarily create a new function with an suffix for the name
    // that later replaces the original function
    mlir::FunctionType ft =
        FunctionType::get(func.getContext(), argTypes, returnTypes);

    std::string funcName =
        (func.getName() + "__rewriting_type_inference_rewriter").str();
    mlir::func::FuncOp newFunc =
        rewriter.create<mlir::func::FuncOp>(func.getLoc(), funcName, ft);

    mapping.map(func.getOperation(), newFunc.getOperation());

    newFunc.addEntryBlock();

    for (auto [oldOperand, newOperand] : llvm::zip_equal(
             func.getBody().getArguments(), newFunc.getBody().getArguments())) {
      mapping.map(oldOperand, newOperand);
    }

    mlir::Block &entryBlock = func.getBody().getBlocks().front();
    mlir::Block &newEntryBlock = newFunc.getBody().getBlocks().front();
    rewriter.setInsertionPointToStart(&newEntryBlock);

    // Recurse into the body of the function
    for (mlir::Operation &op : entryBlock.getOperations()) {
      if (rewrite(&op, rewriter).failed())
        return mlir::failure();
    }

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      if (this->postRewriteHook(rewriter, func, newFunc).failed())
        return mlir::failure();
    }

    // Replace original function and remove suffix from the name of the new
    // function
    rewriter.replaceOp(func, newFunc->getResults());
    newFunc.setName(func.getName());

    return mlir::success();
  }

  mlir::LogicalResult rewrite(mlir::Operation *op, mlir::IRRewriter &rewriter) {
    // Omit type inference debugging operations
    if (llvm::isa<TypeInference::PropagateDownwardOp,
                  TypeInference::PropagateUpwardOp>(op)) {
      mapping.map(op->getResult(0), mapping.lookup(op->getOperand(0)));
      return mlir::success();
    }

    LocalInferenceState resolvedTypes;
    llvm::SmallVector<mlir::Type> resolvedOperandTypes;
    llvm::SmallVector<mlir::Type> resolvedResultTypes;

    // Assemble the list of operands produced by operations that
    // have already been rewritten
    SmallVector<mlir::Value> newOperands = llvm::to_vector(llvm::map_range(
        op->getOperands(), [&](mlir::Value v) { return mapping.lookup(v); }));

    // For operations which do not use unresolved types, simply copy
    // the current types, otherwise resolve types and check that no
    // unresolved types remain
    if (!llvm::any_of(op->getOperands(),
                      [&](mlir::Value v) {
                        return typeResolver.isUnresolvedType(v.getType());
                      }) &&
        !llvm::any_of(op->getResults(), [&](mlir::Value v) {
          return typeResolver.isUnresolvedType(v.getType());
        })) {
      resolvedOperandTypes = llvm::to_vector(op->getOperandTypes());
      resolvedResultTypes = llvm::to_vector(op->getResultTypes());
    } else if (op->hasTrait<mlir::OpTrait::ReturnLike>()) {
      // Return-like ops are a bit special, since they simply forward
      // values upwards. The types inferred for the operands may come
      // from producers, which are in the same block, while the result
      // types of the parent op may have been deduced from producers
      // or consumers of the block containing the parent operation.
      //
      // Blindly taking the operand types may thus result in a
      // mismatch between the final return types of the parent op and
      // the operand types of the return-like op.
      //
      // In the general case, simply take the result types of the
      // parent operation, which, at this point, has already been
      // partially rewritten before recursing into this rewrite call.
      //
      // Functions are a bit diffent though, since the types of the
      // results are contained in a function type and not in the
      // result types.
      mlir::Operation *newParent = mapping.lookup(op->getParentOp());

      if (llvm::isa<mlir::func::ReturnOp>(op)) {
        mlir::func::FuncOp newParentFunc =
            llvm::dyn_cast<mlir::func::FuncOp>(newParent);
        resolvedOperandTypes =
            llvm::to_vector(newParentFunc.getFunctionType().getResults());
      } else {
        // Look up new parent op and use the return types, since these
        // are the authorative types obtained from the last invocation
        // of the type resolver
        resolvedOperandTypes = llvm::to_vector(newParent->getResultTypes());
      }
    } else {
      LocalInferenceState inferredTypes =
          TypeInferenceUtils::getLocalInferenceState(solver, op);

      resolvedTypes = typeResolver.resolve(op, inferredTypes);

      resolvedOperandTypes = resolvedTypes.lookup(op->getOperands());
      resolvedResultTypes = resolvedTypes.lookup(op->getResults());

      if (checkAllResolved(op, resolvedOperandTypes, "operand").failed() ||
          checkAllResolved(op, resolvedResultTypes, "return value").failed()) {
        return mlir::failure();
      }
    }

    // The types retrieved through the last invocation of the type
    // resolver do not necessarily match the types of the operands
    // produced by operations that have already been rewritten, since
    // the resolved types for the producers may originate from
    // iterations of the type inference analysis of different
    // operations.
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      mlir::Type resolvedType = resolvedOperandTypes[i];
      mlir::Type actualType = newOperands[i].getType();

      if (resolvedType != actualType) {
        mlir::OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
        newOperands[i] = this->handleConflict(rewriter, op->getOpOperand(i),
                                              resolvedType, newOperands[i]);
        rewriter.restoreInsertionPoint(ip);
      }
    }

    mlir::Operation *newOp = Operation::create(
        op->getLoc(), op->getName(), resolvedResultTypes, newOperands,
        op->getAttrs(), op->getSuccessors(), op->getNumRegions());

    rewriter.insert(newOp);
    mapping.map(op, newOp);

    // Recurse into nested regions and blocks
    for (size_t regionIdx = 0; regionIdx < op->getNumRegions(); regionIdx++) {
      mlir::Region &oldRegion = op->getRegion(regionIdx);
      mlir::Region &newRegion = newOp->getRegion(regionIdx);

      // Create empty blocks with the right argument types
      for (auto [blockIdx, oldBlock] : llvm::enumerate(oldRegion.getBlocks())) {
        mlir::Block *newBlock = new Block();
        mapping.map(&oldBlock, newBlock);

        for (auto [argIdx, oldArg] : llvm::enumerate(oldBlock.getArguments())) {
          mlir::Type newArgType =
              typeResolver.isUnresolvedType(oldArg.getType())
                  ? resolvedTypes.lookup(oldArg)
                  : oldArg.getType();

          if (!newArgType || typeResolver.isUnresolvedType(newArgType)) {
            op->emitError()
                << "Type of block argument #" << argIdx << " of block #"
                << blockIdx << " of region #" << regionIdx << " is unresolved";
            return mlir::failure();
          }

          mlir::BlockArgument newArg =
              newBlock->addArgument(newArgType, oldArg.getLoc());

          mapping.map(oldArg, newArg);
        }

        newRegion.getBlocks().insert(newRegion.end(), newBlock);
      }

      // Rewrite the contents of the blocks
      for (auto [oldBlock, newBlock] :
           llvm::zip_equal(oldRegion.getBlocks(), newRegion.getBlocks())) {
        rewriter.setInsertionPointToEnd(&newBlock);

        for (mlir::Operation &oldOp : oldBlock)
          if (rewrite(&oldOp, rewriter).failed())
            return mlir::failure();
      }
    }

    rewriter.setInsertionPointAfter(newOp);

    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      mapping.map(oldResult, newResult);
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    return this->postRewriteHook(rewriter, op, newOp);
  }

  // Checks that all types in `types` are resolved and emits an error
  // if this is not the case
  mlir::LogicalResult checkAllResolved(mlir::Operation *op,
                                       mlir::ArrayRef<mlir::Type> types,
                                       const char *kind) {
    for (auto [idx, type] : llvm::enumerate(types)) {
      if (!type || typeResolver.isUnresolvedType(type)) {
        op->emitError() << "Type of " << kind << " #" << idx
                        << " is unresolved";
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  // Hook called when an operation was entirely rewritten. If the hook
  // returns a failure, rewriting of subsequent operations is
  // interrupted and the entire rewrite fails.
  virtual mlir::LogicalResult postRewriteHook(mlir::IRRewriter &rewriter,
                                              mlir::Operation *oldOp,
                                              mlir::Operation *newOp) {
    return mlir::success();
  }

  // Called when an operation is to be created with an operand whose
  // actual type differs from the type that was inferred. By default,
  // this inserts a `TypeInference.unresolved_conflict` op.
  virtual mlir::Value handleConflict(mlir::IRRewriter &rewriter,
                                     mlir::OpOperand &oldOperand,
                                     mlir::Type resolvedType,
                                     mlir::Value producerValue) {
    TypeInference::UnresolvedConflictOp conflict =
        rewriter.create<TypeInference::UnresolvedConflictOp>(
            oldOperand.getOwner()->getLoc(), resolvedType, producerValue);

    return conflict.getResult();
  }

  const mlir::DataFlowSolver &solver;
  TypeResolver &typeResolver;
  mlir::IRMapping mapping;
};

} // namespace concretelang
} // namespace mlir

#endif
