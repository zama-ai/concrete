// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/ADT/SmallVector.h"
#include <concretelang/Transforms/OneShotBufferizeDPSWrapper.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace {
class OneShotBufferizeDPSWrapperPass
    : public OneShotBufferizeDPSWrapperBase<OneShotBufferizeDPSWrapperPass> {
public:
  using OneShotBufferizeDPSWrapperBase<
      OneShotBufferizeDPSWrapperPass>::OneShotBufferizeDPSWrapperBase;

  void runOnOperation() override {
    mlir::MLIRContext *context = &this->getContext();
    mlir::ModuleOp module = this->getOperation();
    mlir::OpBuilder builder(context);

    module.walk([&](mlir::func::FuncOp funcOp) {
      // Skip forward-declarations
      if (funcOp.empty())
        return;

      // Skip functions that do not return vectors
      if (llvm::all_of(funcOp.getFunctionType().getResults(),
                       [](mlir::Type resultTy) {
                         return !resultTy.isa<mlir::TensorType>();
                       }))
        return;

      // Preserve name and type of the original function
      std::string origFuncName = funcOp.getName().str();
      mlir::FunctionType origFuncTy = funcOp.getFunctionType();

      // New input types of the original function: all original inputs
      // plus result memrefs for destination-passing style
      std::vector<mlir::Type> newInputTypes =
          funcOp.getFunctionType().getInputs().vec();

      // New result types of the original function: all original
      // results, except tensor results
      std::vector<mlir::Type> newResultTypes;

      // New function arguments for result memrefs
      std::vector<mlir::Value> newDPSArgs;

      // The result types of the wrapper function: all original
      // results, but tensor results become memrefs
      std::vector<mlir::Type> wrapperResultTypes;

      for (mlir::Type resultTy : funcOp.getFunctionType().getResults()) {
        if (mlir::TensorType tensorResultTy =
                resultTy.dyn_cast<mlir::TensorType>()) {
          mlir::Type memrefResultTy = mlir::MemRefType::get(
              tensorResultTy.getShape(), tensorResultTy.getElementType());
          newInputTypes.push_back(memrefResultTy);
          wrapperResultTypes.push_back(memrefResultTy);

          mlir::Value newDPSArg =
              funcOp.getBody().addArgument(memrefResultTy, funcOp.getLoc());

          newDPSArgs.push_back(newDPSArg);
        } else {
          newResultTypes.push_back(resultTy);
          wrapperResultTypes.push_back(resultTy);
        }
      }

      // Update name and type of the original function
      std::string newFuncName = "__dps_" + origFuncName;
      funcOp.setName(newFuncName);

      mlir::FunctionType newFuncTy =
          mlir::FunctionType::get(context, newInputTypes, newResultTypes);

      funcOp.setType(newFuncTy);

      // Update the terminators of all blocks by extracting all tensor
      // operands, converting them to memrefs, copying their contents
      // to the output memrefs and removing them from the terminator.
      //
      // All non-tensor return values are preserved and returned in
      // the same order.
      for (mlir::Block &block : funcOp.getBlocks()) {
        mlir::Operation *terminator = block.getTerminator();
        builder.setInsertionPoint(terminator);

        size_t newDPSArgIdx = 0;
        size_t operandIdx = 0;

        for (mlir::OpOperand &resOperand : terminator->getOpOperands()) {
          mlir::Value resVal = resOperand.get();

          if (mlir::TensorType resTensorTy =
                  resVal.getType().dyn_cast<mlir::TensorType>()) {

            mlir::Value castedTensor =
                builder.create<mlir::bufferization::ToMemrefOp>(
                    funcOp.getLoc(), newDPSArgs[newDPSArgIdx].getType(),
                    resVal);
            builder.create<mlir::memref::CopyOp>(funcOp.getLoc(), castedTensor,
                                                 newDPSArgs[newDPSArgIdx]);

            newDPSArgIdx++;

            terminator->eraseOperand(operandIdx);
          } else {
            operandIdx++;
          }
        }
      }

      funcOp.setName(newFuncName);

      // Generate wrapper function. The wrapper function allocates
      // memory for each result tensor of the original function and
      // invokes the modified function in destination-passing style
      // with the original arguments plus the output memrefs.
      //
      // The wrapper function returns the results of the original
      // function in the same order, but tensor values are replaced by
      // the output memrefs.
      mlir::FunctionType wrapperFuncTy = mlir::FunctionType::get(
          context, origFuncTy.getInputs(), wrapperResultTypes);

      builder.setInsertionPoint(funcOp);

      mlir::func::FuncOp wrapperFuncOp = builder.create<mlir::func::FuncOp>(
          funcOp.getLoc(), origFuncName, wrapperFuncTy,
          builder.getStringAttr("private"));

      mlir::Block *wrapperEntryBlock = wrapperFuncOp.addEntryBlock();

      // Generate call of the original function in destination-passing
      // style
      builder.setInsertionPointToStart(wrapperEntryBlock);
      mlir::func::CallOp callOp =
          builder.create<mlir::func::CallOp>(funcOp.getLoc(), funcOp);
      builder.create<mlir::func::ReturnOp>(funcOp.getLoc());

      mlir::Operation *wrapperTerminator =
          wrapperFuncOp.getBody().getBlocks().front().getTerminator();

      // Create allocations of the result memrefs in the wrapper
      // function and create arguments for the call operation invoking
      // the original function in destination-passing style
      callOp.getOperation()->setOperands(wrapperFuncOp.getArguments());
      builder.setInsertionPoint(callOp);

      size_t callArgIndex = callOp.getOperation()->getNumOperands();
      llvm::SmallVector<mlir::Value, 1> dpsResultValues;

      // Allocate the output memrefs and add to the end of operands to
      // the call po inviking the modified function in
      // destination-passing style
      for (mlir::Value newDPSArg : newDPSArgs) {
        mlir::MemRefType memrefTy =
            newDPSArg.getType().dyn_cast<mlir::MemRefType>();

        mlir::memref::AllocOp allocOp =
            builder.create<mlir::memref::AllocOp>(funcOp.getLoc(), memrefTy);
        dpsResultValues.push_back(allocOp.getResult());
        callOp.getOperation()->insertOperands(callArgIndex,
                                              allocOp.getResult());
        callArgIndex++;
      }

      // Build up the list of operands of the wrapper function,
      // composed of the return values of the modified function and
      // the memrefs containing the poutput values after invocation of
      // the modified function in destination-passing style
      size_t dpsResultIndex = 0;
      size_t resultIndex = 0;
      size_t origResultIndex = 0;
      for (mlir::Type origResultTy : origFuncTy.getResults()) {
        if (origResultTy.isa<mlir::TensorType>()) {
          wrapperTerminator->insertOperands(resultIndex,
                                            dpsResultValues[dpsResultIndex]);
          dpsResultIndex++;
        } else {
          wrapperTerminator->insertOperands(resultIndex,
                                            callOp.getResult(origResultIndex));
          origResultIndex++;
        }

        resultIndex++;
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::concretelang::createOneShotBufferizeDPSWrapperPass() {
  return std::make_unique<OneShotBufferizeDPSWrapperPass>();
}
