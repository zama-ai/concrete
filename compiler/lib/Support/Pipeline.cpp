#include <llvm/Support/TargetSelect.h>

#include <llvm/Support/Error.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <zamalang/Conversion/Passes.h>
#include <zamalang/Dialect/HLFHE/Analysis/MANP.h>
#include <zamalang/Support/Pipeline.h>
#include <zamalang/Support/logging.h>
#include <zamalang/Support/math.h>

namespace mlir {
namespace zamalang {
namespace pipeline {
static void addPotentiallyNestedPass(mlir::PassManager &pm,
                                     std::unique_ptr<Pass> pass) {
  if (!pass->getOpName() || *pass->getOpName() == "builtin.module") {
    pm.addPass(std::move(pass));
  } else {
    pm.nest(*pass->getOpName()).addPass(std::move(pass));
  }
}

// Creates an instance of the Minimal Arithmetic Noise Padding pass
// and invokes it for all functions of `module`.
mlir::LogicalResult invokeMANPPass(mlir::MLIRContext &context,
                                   mlir::ModuleOp &module, bool debug) {
  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::FuncOp>(mlir::zamalang::createMANPPass(debug));
  return pm.run(module);
}

llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
getFHEConstraintsFromHLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module) {
  llvm::Optional<size_t> oMax2norm;
  llvm::Optional<size_t> oMaxWidth;

  mlir::PassManager pm(&context);

  addPotentiallyNestedPass(pm, mlir::zamalang::createMANPPass());
  addPotentiallyNestedPass(
      pm, mlir::zamalang::createMaxMANPPass([&](const llvm::APInt &currMaxMANP,
                                                unsigned currMaxWidth) {
        assert((uint64_t)currMaxWidth < std::numeric_limits<size_t>::max() &&
               "Maximum width does not fit into size_t");

        assert(sizeof(uint64_t) >= sizeof(size_t) &&
               currMaxMANP.ult(std::numeric_limits<size_t>::max()) &&
               "Maximum MANP does not fit into size_t");

        size_t manp = (size_t)currMaxMANP.getZExtValue();
        size_t width = (size_t)currMaxWidth;

        if (!oMax2norm.hasValue() || oMax2norm.getValue() < manp)
          oMax2norm.emplace(manp);

        if (!oMaxWidth.hasValue() || oMaxWidth.getValue() < width)
          oMaxWidth.emplace(width);
      }));

  if (pm.run(module.getOperation()).failed()) {
    return llvm::make_error<llvm::StringError>(
        "Failed to determine the maximum Arithmetic Noise Padding and maximum"
        "required precision",
        llvm::inconvertibleErrorCode());
  }

  llvm::Optional<mlir::zamalang::V0FHEConstraint> ret;

  if (oMax2norm.hasValue() && oMaxWidth.hasValue()) {
    ret = llvm::Optional<mlir::zamalang::V0FHEConstraint>(
        {.norm2 = ceilLog2(oMax2norm.getValue()), .p = oMaxWidth.getValue()});
  }

  return ret;
}

mlir::LogicalResult lowerHLFHEToMidLFHE(mlir::MLIRContext &context,
                                        mlir::ModuleOp &module, bool verbose) {
  mlir::PassManager pm(&context);

  if (verbose) {
    mlir::zamalang::log_verbose()
        << "##################################################\n"
        << "### HLFHE to MidLFHE pipeline\n";

    pm.enableIRPrinting();
    pm.enableStatistics();
    pm.enableTiming();
    pm.enableVerifier();
  }

  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertHLFHETensorOpsToLinalg());
  addPotentiallyNestedPass(pm,
                           mlir::zamalang::createConvertHLFHEToMidLFHEPass());

  return pm.run(module.getOperation());
}

mlir::LogicalResult lowerMidLFHEToLowLFHE(mlir::MLIRContext &context,
                                          mlir::ModuleOp &module,
                                          V0FHEContext &fheContext,
                                          bool parametrize) {
  mlir::PassManager pm(&context);

  if (parametrize) {
    addPotentiallyNestedPass(
        pm, mlir::zamalang::createConvertMidLFHEGlobalParametrizationPass(
                fheContext));
  }

  addPotentiallyNestedPass(pm,
                           mlir::zamalang::createConvertMidLFHEToLowLFHEPass());

  return pm.run(module.getOperation());
}

mlir::LogicalResult lowerLowLFHEToStd(mlir::MLIRContext &context,
                                      mlir::ModuleOp &module) {
  mlir::PassManager pm(&context);
  pm.addPass(mlir::zamalang::createConvertLowLFHEToConcreteCAPIPass());
  return pm.run(module.getOperation());
}

mlir::LogicalResult lowerStdToLLVMDialect(mlir::MLIRContext &context,
                                          mlir::ModuleOp &module,
                                          bool verbose) {
  mlir::PassManager pm(&context);

  if (verbose) {
    mlir::zamalang::log_verbose()
        << "##################################################\n"
        << "### MlirStdsDialectToMlirLLVMDialect pipeline\n";
    context.disableMultithreading();
    pm.enableIRPrinting();
    pm.enableStatistics();
    pm.enableTiming();
    pm.enableVerifier();
  }

  // Unparametrize LowLFHE
  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertLowLFHEUnparametrizePass());

  // Bufferize
  addPotentiallyNestedPass(pm, mlir::createTensorConstantBufferizePass());
  addPotentiallyNestedPass(pm, mlir::createStdBufferizePass());
  addPotentiallyNestedPass(pm, mlir::createTensorBufferizePass());
  addPotentiallyNestedPass(pm, mlir::createLinalgBufferizePass());
  addPotentiallyNestedPass(pm, mlir::createConvertLinalgToLoopsPass());
  addPotentiallyNestedPass(pm, mlir::createFuncBufferizePass());
  addPotentiallyNestedPass(pm, mlir::createFinalizingBufferizePass());

  // Convert to MLIR LLVM Dialect
  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertMLIRLowerableDialectsToLLVMPass());

  return pm.run(module);
}

std::unique_ptr<llvm::Module>
lowerLLVMDialectToLLVMIR(mlir::MLIRContext &context,
                         llvm::LLVMContext &llvmContext,
                         mlir::ModuleOp &module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*module->getContext());

  return mlir::translateModuleToLLVMIR(module, llvmContext);
}

mlir::LogicalResult optimizeLLVMModule(llvm::LLVMContext &llvmContext,
                                       llvm::Module &module) {
  std::function<llvm::Error(llvm::Module *)> optPipeline =
      mlir::makeOptimizingTransformer(3, 0, nullptr);

  if (optPipeline(&module))
    return mlir::failure();
  else
    return mlir::success();
}

mlir::LogicalResult lowerHLFHEToStd(mlir::MLIRContext &context,
                                    mlir::ModuleOp &module,
                                    V0FHEContext &fheContext, bool verbose) {
  if (lowerHLFHEToMidLFHE(context, module, verbose).failed() ||
      lowerMidLFHEToLowLFHE(context, module, fheContext, true).failed() ||
      lowerLowLFHEToStd(context, module).failed()) {
    return mlir::failure();
  } else {
    return mlir::success();
  }
}

} // namespace pipeline
} // namespace zamalang
} // namespace mlir
