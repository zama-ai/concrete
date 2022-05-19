// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <llvm/Support/TargetSelect.h>

#include <llvm/Support/Error.h>
#include <mlir/Conversion/Passes.h>

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/SCF/Passes.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <concretelang/Conversion/Passes.h>
#include <concretelang/Dialect/Concrete/Transforms/Optimization.h>
#include <concretelang/Dialect/FHE/Analysis/MANP.h>
#include <concretelang/Dialect/FHELinalg/Transforms/Tiling.h>
#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Support/Pipeline.h>
#include <concretelang/Support/logging.h>
#include <concretelang/Support/math.h>
#include <concretelang/Transforms/Bufferize.h>

namespace mlir {
namespace concretelang {
namespace pipeline {

static void pipelinePrinting(llvm::StringRef name, mlir::PassManager &pm,
                             mlir::MLIRContext &ctx) {
  if (mlir::concretelang::isVerbose()) {
    mlir::concretelang::log_verbose()
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

llvm::Expected<llvm::Optional<mlir::concretelang::V0FHEConstraint>>
getFHEConstraintsFromFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                         std::function<bool(mlir::Pass *)> enablePass) {
  llvm::Optional<size_t> oMax2norm;
  llvm::Optional<size_t> oMaxWidth;

  mlir::PassManager pm(&context);

  pipelinePrinting("ComputeFHEConstraintOnFHE", pm, context);
  addPotentiallyNestedPass(pm, mlir::createCanonicalizerPass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::concretelang::createMANPPass(),
                           enablePass);
  addPotentiallyNestedPass(
      pm,
      mlir::concretelang::createMaxMANPPass([&](const llvm::APInt &currMaxMANP,
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
  llvm::Optional<mlir::concretelang::V0FHEConstraint> ret;

  if (oMax2norm.hasValue() && oMaxWidth.hasValue()) {
    ret = llvm::Optional<mlir::concretelang::V0FHEConstraint>(
        {/*.norm2 = */ ceilLog2(oMax2norm.getValue()),
         /*.p = */ oMaxWidth.getValue()});
  }

  return ret;
}

mlir::LogicalResult autopar(mlir::MLIRContext &context, mlir::ModuleOp &module,
                            std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("AutoPar", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createBuildDataflowTaskGraphPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
tileMarkedFHELinalg(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TileMarkedFHELinalg", pm, context);
  addPotentiallyNestedPass(pm, mlir::concretelang::createFHELinalgTilingPass(),
                           enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
markFHELinalgForTiling(mlir::MLIRContext &context, mlir::ModuleOp &module,
                       llvm::ArrayRef<int64_t> tileSizes,
                       std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("MarkFHELinalgForTiling", pm, context);
  addPotentiallyNestedPass(pm, createFHELinalgTilingMarkerPass(tileSizes),
                           enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerFHEToTFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
               std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("FHEToTFHE", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertFHETensorOpsToLinalg(), enablePass);
  // FHETensorOpsToLinalg does generate linalg named ops that need to be lowered
  // to linalg.generic operations
  addPotentiallyNestedPass(pm, mlir::createLinalgGeneralizationPass(),
                           enablePass);
  addPotentiallyNestedPass(pm, mlir::concretelang::createConvertFHEToTFHEPass(),
                           enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerTFHEToConcrete(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    llvm::Optional<V0FHEContext> &fheContext,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TFHEToConcrete", pm, context);

  if (fheContext.hasValue()) {
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createConvertTFHEGlobalParametrizationPass(
            fheContext.getValue()),
        enablePass);
  }

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertTFHEToConcretePass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
optimizeConcrete(mlir::MLIRContext &context, mlir::ModuleOp &module,
                 std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("ConcreteOptimization", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConcreteOptimizationPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerConcreteToBConcrete(mlir::MLIRContext &context, mlir::ModuleOp &module,
                         std::function<bool(mlir::Pass *)> enablePass,
                         bool parallelizeLoops) {
  mlir::PassManager pm(&context);
  pipelinePrinting("ConcreteToBConcrete", pm, context);
  addPotentiallyNestedPass(
      pm,
      mlir::concretelang::createConvertConcreteToBConcretePass(
          parallelizeLoops),
      enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerBConcreteToStd(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("BConcreteToStd", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertBConcreteToBConcreteCAPIPass(),
      enablePass);
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerStdToLLVMDialect(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass,
                      bool parallelizeLoops) {
  mlir::PassManager pm(&context);
  pipelinePrinting("StdToLLVM", pm, context);

  // Bufferize
  addPotentiallyNestedPass(pm, mlir::createTensorConstantBufferizePass(),
                           enablePass);
  addPotentiallyNestedPass(pm, mlir::createStdBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createTensorBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createLinalgBufferizePass(), enablePass);
  if (parallelizeLoops)
    addPotentiallyNestedPass(pm, mlir::createConvertLinalgToParallelLoopsPass(),
                             enablePass);
  else
    addPotentiallyNestedPass(pm, mlir::createConvertLinalgToLoopsPass(),
                             enablePass);
  addPotentiallyNestedPass(pm, mlir::createSCFBufferizePass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createFuncBufferizePass(), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createBufferizeDataflowTaskOpsPass(), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createFinalizingBufferizePass(), enablePass);

  addPotentiallyNestedPass(pm, mlir::createBufferDeallocationPass(),
                           enablePass);

  if (parallelizeLoops)
    addPotentiallyNestedPass(pm, mlir::createConvertSCFToOpenMPPass(),
                             enablePass);
  // Lower affine
  addPotentiallyNestedPass(pm, mlir::createLowerAffinePass(), enablePass);

  // Lower Dataflow tasks to DRF
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createFixupDataflowTaskOpsPass(), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createLowerDataflowTasksPass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createLowerToCFGPass(), enablePass);

  // Convert to MLIR LLVM Dialect
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertMLIRLowerableDialectsToLLVMPass(),
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
  mlir::registerOpenMPDialectTranslation(*module->getContext());

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

} // namespace pipeline
} // namespace concretelang
} // namespace mlir
