// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <llvm/Support/TargetSelect.h>

#include <llvm/Support/Error.h>
#include <mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include <concretelang/Conversion/Passes.h>
#include <concretelang/Dialect/BConcrete/Transforms/Passes.h>
#include <concretelang/Dialect/Concrete/Transforms/Optimization.h>
#include <concretelang/Dialect/FHE/Analysis/ConcreteOptimizer.h>
#include <concretelang/Dialect/FHE/Analysis/MANP.h>
#include <concretelang/Dialect/FHE/Transforms/Boolean.h>
#include <concretelang/Dialect/FHE/Transforms/EncryptedMulToDoubleTLU.h>
#include <concretelang/Dialect/FHELinalg/Transforms/Tiling.h>
#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Support/Pipeline.h>
#include <concretelang/Support/logging.h>
#include <concretelang/Support/math.h>
#include <concretelang/Transforms/Passes.h>

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

llvm::Expected<std::map<std::string, llvm::Optional<optimizer::Description>>>
getFHEContextFromFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                     optimizer::Config config,
                     std::function<bool(mlir::Pass *)> enablePass) {
  llvm::Optional<size_t> oMax2norm;
  llvm::Optional<size_t> oMaxWidth;
  optimizer::FunctionsDag dags;

  mlir::PassManager pm(&context);

  pipelinePrinting("ComputeFHEConstraintOnFHE", pm, context);
  addPotentiallyNestedPass(pm, mlir::createCanonicalizerPass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::concretelang::createMANPPass(),
                           enablePass);
  addPotentiallyNestedPass(
      pm,
      mlir::concretelang::createMaxMANPPass(
          [&](const uint64_t manp, unsigned width) {
            if (!oMax2norm.hasValue() || oMax2norm.getValue() < manp)
              oMax2norm.emplace(manp);

            if (!oMaxWidth.hasValue() || oMaxWidth.getValue() < width)
              oMaxWidth.emplace(width);
          }),
      enablePass);
  if (pm.run(module.getOperation()).failed()) {
    return llvm::make_error<llvm::StringError>(
        "Failed to determine the maximum Arithmetic Noise Padding and maximum"
        " required precision",
        llvm::inconvertibleErrorCode());
  }
  llvm::Optional<mlir::concretelang::V0FHEConstraint> constraint = llvm::None;

  if (oMax2norm.hasValue() && oMaxWidth.hasValue()) {
    constraint = llvm::Optional<mlir::concretelang::V0FHEConstraint>(
        {/*.norm2 = */ ceilLog2(oMax2norm.getValue()),
         /*.p = */ oMaxWidth.getValue()});
  }
  addPotentiallyNestedPass(pm, optimizer::createDagPass(config, dags),
                           enablePass);
  if (pm.run(module.getOperation()).failed()) {
    return StreamStringError() << "Failed to create concrete-optimizer dag\n";
  }
  std::map<std::string, llvm::Optional<optimizer::Description>> descriptions;
  for (auto &entry_dag : dags) {
    if (!constraint) {
      descriptions.insert(
          decltype(descriptions)::value_type(entry_dag.first, llvm::None));
      continue;
    }
    optimizer::Description description = {*constraint,
                                          std::move(entry_dag.second)};
    llvm::Optional<optimizer::Description> opt_description{
        std::move(description)};
    descriptions.insert(decltype(descriptions)::value_type(
        entry_dag.first, std::move(opt_description)));
  }
  return std::move(descriptions);
}

mlir::LogicalResult autopar(mlir::MLIRContext &context, mlir::ModuleOp &module,
                            std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("AutoPar", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createBuildDataflowTaskGraphPass(), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createLowerDataflowTasksPass(), enablePass);

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
transformHighLevelFHEOps(mlir::MLIRContext &context, mlir::ModuleOp &module,
                         std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("transformHighLevelFHEOps", pm, context);
  addPotentiallyNestedPass(pm, createEncryptedMulToDoubleTLUPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerFHELinalgToFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    llvm::Optional<V0FHEContext> &fheContext,
                    std::function<bool(mlir::Pass *)> enablePass,
                    bool parallelizeLoops, bool batchOperations) {
  mlir::PassManager pm(&context);
  pipelinePrinting("FHELinalgToFHE", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertFHETensorOpsToLinalg(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createLinalgGeneralizationPass(),
                           enablePass);
  addPotentiallyNestedPass(
      pm,
      mlir::concretelang::createLinalgGenericOpWithTensorsToLoopsPass(
          parallelizeLoops),
      enablePass);

  if (batchOperations) {
    addPotentiallyNestedPass(pm, mlir::concretelang::createBatchingPass(),
                             enablePass);
  }
  return pm.run(module.getOperation());
}

mlir::LogicalResult
transformFHEBoolean(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createFHEBooleanTransformPass(), enablePass);
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerFHEToTFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
               llvm::Optional<V0FHEContext> &fheContext,
               std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);

  if (fheContext.hasValue() && fheContext->parameter.largeInteger.hasValue()) {
    pipelinePrinting("FHEToTFHECrt", pm, context);
    auto dec =
        fheContext.value().parameter.largeInteger.value().crtDecomposition;
    auto mods = mlir::SmallVector<int64_t>(dec.begin(), dec.end());
    auto polySize = fheContext.value().parameter.getPolynomialSize();
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createConvertFHEToTFHECrtPass(
            mlir::concretelang::CrtLoweringParameters(mods, polySize)),
        enablePass);
  } else if (fheContext.hasValue()) {
    pipelinePrinting("FHEToTFHEScalar", pm, context);
    size_t polySize = fheContext.value().parameter.getPolynomialSize();
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createConvertFHEToTFHEScalarPass(
            mlir::concretelang::ScalarLoweringParameters(polySize)),
        enablePass);
  }

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
      pm, mlir::concretelang::createConvertConcreteToBConcretePass(),
      enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult extractSDFGOps(mlir::MLIRContext &context,
                                   mlir::ModuleOp &module,
                                   std::function<bool(mlir::Pass *)> enablePass,
                                   bool unroll) {
  mlir::PassManager pm(&context);
  pipelinePrinting("extract SDFG ops from BConcrete", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createExtractSDFGOpsPass(unroll), enablePass);
  LogicalResult res = pm.run(module.getOperation());

  return res;
}

mlir::LogicalResult
lowerBConcreteToStd(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("BConcreteToStd", pm, context);
  addPotentiallyNestedPass(pm, mlir::concretelang::createAddRuntimeContext(),
                           enablePass);
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerSDFGToStd(mlir::MLIRContext &context, mlir::ModuleOp &module,
               std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("SDFGToStd", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertSDFGToStreamEmulatorPass(),
      enablePass);
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerStdToLLVMDialect(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass,
                      bool parallelizeLoops, bool gpu) {
  mlir::PassManager pm(&context);
  pipelinePrinting("StdToLLVM", pm, context);

  // Bufferize
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.allowReturnAllocs = true;
  bufferizationOptions.printConflicts = true;
  bufferizationOptions.unknownTypeConversion = mlir::bufferization::
      OneShotBufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  bufferizationOptions.createDeallocs = true;

  std::unique_ptr<mlir::Pass> comprBuffPass =
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions);

  addPotentiallyNestedPass(pm, std::move(comprBuffPass), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createBufferizeDataflowTaskOpsPass(), enablePass);

  if (parallelizeLoops) {
    addPotentiallyNestedPass(
        pm, mlir::concretelang::createCollapseParallelLoops(), enablePass);
    addPotentiallyNestedPass(pm, mlir::concretelang::createForLoopToParallel(),
                             enablePass);
  }

  if (parallelizeLoops)
    addPotentiallyNestedPass(pm, mlir::createConvertSCFToOpenMPPass(),
                             enablePass);
  // Lower affine
  addPotentiallyNestedPass(pm, mlir::createLowerAffinePass(), enablePass);

  // Finalize the lowering of RT/DFR which includes:
  //   - adding type and typesize information for dependences
  //   - issue _dfr_start and _dfr_stop calls to start/stop the runtime
  //   - remove deallocation calls for buffers managed through refcounting
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createFinalizeTaskCreationPass(), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::bufferization::createBufferDeallocationPass(), enablePass);
  addPotentiallyNestedPass(pm, mlir::concretelang::createStartStopPass(),
                           enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createFixupBufferDeallocationPass(), enablePass);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertBConcreteToCAPIPass(gpu),
      enablePass);

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
