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

    // Re-write temporary function names and erase old functions that
    // have been rewritten
    llvm::SmallVector<mlir::func::FuncOp> funcsToErase;

    for (auto [fromOp, toOp] : mapping.getOperationMap()) {
      mlir::func::FuncOp fromFunc = llvm::dyn_cast<mlir::func::FuncOp>(fromOp);
      mlir::func::FuncOp toFunc = llvm::dyn_cast<mlir::func::FuncOp>(toOp);

      if (fromFunc && toFunc) {
        std::string fromFuncName = fromFunc.getName().str();
        toFunc.setName(fromFuncName);
        funcsToErase.push_back(fromFunc);
      }
    }

    for (mlir::func::FuncOp funcToErase : funcsToErase) {
      funcToErase.erase();
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    return this->postRewriteHook(rewriter, module, module);
  }

protected:
  // Rewrites all the blocks of all regions of `srcOp` into `tgtOp`
  mlir::LogicalResult rewriteRegions(mlir::Operation *srcOp,
                                     mlir::Operation *tgtOp,
                                     const LocalInferenceState &resolvedTypes,
                                     mlir::IRRewriter &rewriter) {
    for (size_t regionIdx = 0; regionIdx < srcOp->getNumRegions();
         regionIdx++) {
      mlir::Region &oldRegion = srcOp->getRegion(regionIdx);
      mlir::Region &newRegion = tgtOp->getRegion(regionIdx);

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
            srcOp->emitError()
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

    return mlir::success();
  }

  // Rewrite a single function
  mlir::LogicalResult rewrite(mlir::func::FuncOp func,
                              mlir::IRRewriter &rewriter) {
    rewriter.setInsertionPointAfter(func);

    // Gather inferred types an run the solver a last time on the
    // function to get the actual types for the arguments and return
    // values
    LocalInferenceState inferredTypes =
        TypeInferenceUtils::getLocalInferenceState(solver, typeResolver, func);
    LocalInferenceState resolvedTypes =
        typeResolver.resolve(func, inferredTypes);

    llvm::SmallVector<mlir::Type> argTypes =
        resolvedTypes.lookup(func.getBody().getArguments());

    // Check that all argument types have been resolved
    if (checkAllResolved(func, argTypes, "argument").failed()) {
      return mlir::failure();
    }

    std::optional<llvm::SmallVector<mlir::Type>> returnTypes;

    // Check that the types of the values returned by all func.return
    // terminators have the same types
    for (auto &[i, block] : llvm::enumerate(func.getRegion().getBlocks())) {
      if (mlir::func::ReturnOp returnOp =
              llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator())) {
        if (!returnTypes.has_value()) {
          returnTypes = resolvedTypes.lookup(returnOp->getOperands());

          // Check that the types of the values returned through first
          // terminator have all been resolved
          if (checkAllResolved(returnOp, *returnTypes, "argument").failed()) {
            return mlir::failure();
          }
        } else {
          for (auto [j, operand] : llvm::enumerate(returnOp->getOperands())) {
            mlir::Type thisOperandType = resolvedTypes.lookup(operand);
            mlir::Type firstReturnOperand = (*returnTypes)[j];

            if (thisOperandType != firstReturnOperand) {
              func->emitError("Resolved type ")
                  << thisOperandType << " for operand #" << i
                  << " of terminator #" << j << " differs from the type "
                  << firstReturnOperand
                  << " of the same operand of the first "
                     "return operation";
            }
          }
        }
      }
    }

    // Temporarily create a new function with an suffix for the name
    // that later replaces the original function
    mlir::FunctionType ft =
        FunctionType::get(func.getContext(), argTypes, *returnTypes);

    std::string funcName =
        (func.getName() + "__rewriting_type_inference_rewriter").str();
    mlir::func::FuncOp newFunc =
        rewriter.create<mlir::func::FuncOp>(func.getLoc(), funcName, ft);

    mapping.map(func.getOperation(), newFunc.getOperation());

    // Recurse into nested regions and blocks
    if (rewriteRegions(func, newFunc, resolvedTypes, rewriter).failed())
      return mlir::failure();

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      if (this->postRewriteHook(rewriter, func, newFunc).failed())
        return mlir::failure();
    }

    // Replace results of the original function, but do not yet erase
    // it, since it may still be referenced in functions that have not
    // been rewritten, yet
    func->replaceAllUsesWith(newFunc->getResults());

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
    } else {
      LocalInferenceState inferredTypes =
          TypeInferenceUtils::getLocalInferenceState(solver, typeResolver, op);

      // Return-like ops are a bit special, since their operand types
      // are tied to the types of their parent op. The types inferred
      // for the operands of a return-like operation may come from
      // producers, which are in the same block, while the result
      // types of the parent op may have been deduced from producers
      // or consumers of the block containing the parent operation.
      //
      // Blindly applying the inferred operand types may thus result
      // in a mismatch between the final return types of the parent op
      // and the operand types of the return-like op.
      //
      // Instead, look up the rewritten parent op and add its return
      // types to the local inference state before invoking type
      // inference a last time.
      if (op->hasTrait<mlir::OpTrait::ReturnLike>()) {
        mlir::Operation *newParent = mapping.lookup(op->getParentOp());

        for (auto [oldResult, newResult] : llvm::zip_equal(
                 op->getParentOp()->getResults(), newParent->getResults())) {
          inferredTypes.set(oldResult, newResult.getType());
        }
      }

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

    SmallVector<Block *> newSuccessors;
    for (Block *successor : op->getSuccessors()) {
      newSuccessors.push_back(mapping.lookupOrDefault(successor));
    }

    mlir::Operation *newOp = Operation::create(
        op->getLoc(), op->getName(), resolvedResultTypes, newOperands,
        op->getAttrs(), newSuccessors, op->getNumRegions());

    rewriter.insert(newOp);
    mapping.map(op, newOp);

    // Recurse into nested regions and blocks
    if (rewriteRegions(op, newOp, resolvedTypes, rewriter).failed())
      return mlir::failure();

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
