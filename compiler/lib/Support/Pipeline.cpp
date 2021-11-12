#include <llvm/Support/TargetSelect.h>

#include <llvm/Support/Error.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
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

static void pipelinePrinting(llvm::StringRef name, mlir::PassManager &pm,
                             mlir::MLIRContext &ctx) {
  if (mlir::zamalang::isVerbose()) {
    mlir::zamalang::log_verbose()
        << "##################################################\n"
        << "### " << name << " pipeline\n";
    auto isModule = [](mlir::Pass *, mlir::Operation *op) {
      return mlir::isa<mlir::ModuleOp>(op);
    };
    ctx.disableMultithreading(true);
    pm.enableIRPrinting(isModule, isModule);
    pm.enableStatistics();
    pm.enableTiming();
    pm.enableVerifier();
  }
}

static void
addPotentiallyNestedPass(mlir::PassManager &pm, std::unique_ptr<Pass> pass,
                         std::function<bool(mlir::Pass *)> enablePass) {
  if (!enablePass(pass.get())) {
    return;
  }
  if (!pass->getOpName() || *pass->getOpName() == "builtin.module") {
    pm.addPass(std::move(pass));
  } else {
    mlir::OpPassManager &p = pm.nest(*pass->getOpName());
    p.addPass(std::move(pass));
  }
}

llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
getFHEConstraintsFromHLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                           std::function<bool(mlir::Pass *)> enablePass) {
  llvm::Optional<size_t> oMax2norm;
  llvm::Optional<size_t> oMaxWidth;

  mlir::PassManager pm(&context);

  pipelinePrinting("ComputeFHEConstraintOnHLFHE", pm, context);
  addPotentiallyNestedPass(pm, mlir::zamalang::createMANPPass(), enablePass);
  addPotentiallyNestedPass(
      pm,
      mlir::zamalang::createMaxMANPPass([&](const llvm::APInt &currMaxMANP,
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
      }),
      enablePass);
  if (pm.run(module.getOperation()).failed()) {
    return llvm::make_error<llvm::StringError>(
        "Failed to determine the maximum Arithmetic Noise Padding and maximum"
        "required precision",
        llvm::inconvertibleErrorCode());
  }
  llvm::Optional<mlir::zamalang::V0FHEConstraint> ret;

  if (oMax2norm.hasValue() && oMaxWidth.hasValue()) {
    ret = llvm::Optional<mlir::zamalang::V0FHEConstraint>(
        {/*.norm2 = */ ceilLog2(oMax2norm.getValue()),
         /*.p = */ oMaxWidth.getValue()});
  }

  return ret;
}

mlir::LogicalResult
lowerHLFHEToMidLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("HLFHEToMidLFHE", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertHLFHETensorOpsToLinalg(), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertHLFHEToMidLFHEPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerMidLFHEToLowLFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      llvm::Optional<V0FHEContext> &fheContext,
                      std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("MidLFHEToLowLFHE", pm, context);

  if (fheContext.hasValue()) {
    addPotentiallyNestedPass(
        pm,
        mlir::zamalang::createConvertMidLFHEGlobalParametrizationPass(
            fheContext.getValue()),
        enablePass);
  }

  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertMidLFHEToLowLFHEPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerLowLFHEToStd(mlir::MLIRContext &context, mlir::ModuleOp &module,
                  std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("LowLFHEToStd", pm, context);
  pm.addPass(mlir::zamalang::createConvertLowLFHEToConcreteCAPIPass());
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerStdToLLVMDialect(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("StdToLLVM", pm, context);

  // Unparametrize LowLFHE
  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertLowLFHEUnparametrizePass(), enablePass);

  // Bufferize
  addPotentiallyNestedPass(pm, mlir::createTensorConstantBufferizePass(),
                           enablePass);
  addPotentiallyNestedPass(pm, mlir::createStdBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createTensorBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createLinalgBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createConvertLinalgToLoopsPass(),
                           enablePass);
  addPotentiallyNestedPass(pm, mlir::createFuncBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createFinalizingBufferizePass(),
                           enablePass);
  addPotentiallyNestedPass(pm, mlir::createLowerToCFGPass(), enablePass);

  // Convert to MLIR LLVM Dialect
  addPotentiallyNestedPass(
      pm, mlir::zamalang::createConvertMLIRLowerableDialectsToLLVMPass(),
      enablePass);

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
  llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline =
      mlir::makeOptimizingTransformer(3, 0, nullptr);

  if (optPipeline(&module))
    return mlir::failure();
  else
    return mlir::success();
}

} // namespace pipeline
} // namespace zamalang
} // namespace mlir
