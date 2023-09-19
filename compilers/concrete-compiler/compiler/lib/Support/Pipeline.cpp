// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/Support/TargetSelect.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Error.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/TFHEKeyNormalization/Pass.h"
#include "concretelang/Dialect/Concrete/Analysis/MemoryUsage.h"
#include "concretelang/Dialect/Concrete/Transforms/Passes.h"
#include "concretelang/Dialect/FHE/Analysis/ConcreteOptimizer.h"
#include "concretelang/Dialect/FHE/Analysis/MANP.h"
#include "concretelang/Dialect/FHE/Transforms/BigInt/BigInt.h"
#include "concretelang/Dialect/FHE/Transforms/Boolean/Boolean.h"
#include "concretelang/Dialect/FHE/Transforms/DynamicTLU/DynamicTLU.h"
#include "concretelang/Dialect/FHE/Transforms/EncryptedMulToDoubleTLU/EncryptedMulToDoubleTLU.h"
#include "concretelang/Dialect/FHE/Transforms/Max/Max.h"
#include "concretelang/Dialect/FHELinalg/Transforms/Tiling.h"
#include "concretelang/Dialect/RT/Analysis/Autopar.h"
#include "concretelang/Dialect/TFHE/Analysis/ExtractStatistics.h"
#include "concretelang/Dialect/TFHE/Transforms/Transforms.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/Pipeline.h"
#include "concretelang/Support/logging.h"
#include "concretelang/Support/math.h"
#include "concretelang/Transforms/Passes.h"

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

llvm::Expected<std::map<std::string, std::optional<optimizer::Description>>>
getFHEContextFromFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                     optimizer::Config config,
                     std::function<bool(mlir::Pass *)> enablePass) {
  std::optional<size_t> oMax2norm;
  std::optional<size_t> oMaxWidth;
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
            if (!oMax2norm.has_value() || oMax2norm.value() < manp)
              oMax2norm.emplace(manp);

            if (!oMaxWidth.has_value() || oMaxWidth.value() < width)
              oMaxWidth.emplace(width);
          }),
      enablePass);
  if (pm.run(module.getOperation()).failed()) {
    return llvm::make_error<llvm::StringError>(
        "Failed to determine the maximum Arithmetic Noise Padding and maximum"
        " required precision",
        llvm::inconvertibleErrorCode());
  }
  std::optional<mlir::concretelang::V0FHEConstraint> constraint = std::nullopt;

  if (oMax2norm.has_value() && oMaxWidth.has_value()) {
    constraint = std::optional<mlir::concretelang::V0FHEConstraint>(
        {/*.norm2 = */ ceilLog2(oMax2norm.value()),
         /*.p = */ oMaxWidth.value()});
  }
  addPotentiallyNestedPass(pm, optimizer::createDagPass(config, dags),
                           enablePass);
  if (pm.run(module.getOperation()).failed()) {
    return StreamStringError() << "Failed to create concrete-optimizer dag\n";
  }
  std::map<std::string, std::optional<optimizer::Description>> descriptions;
  for (auto &entry_dag : dags) {
    if (!constraint) {
      descriptions.insert(
          decltype(descriptions)::value_type(entry_dag.first, std::nullopt));
      continue;
    }
    optimizer::Description description = {*constraint,
                                          std::move(entry_dag.second)};
    std::optional<optimizer::Description> opt_description{
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
tileMarkedLinalg(mlir::MLIRContext &context, mlir::ModuleOp &module,
                 std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TileMarkedLinalg", pm, context);
  addPotentiallyNestedPass(pm, mlir::concretelang::createLinalgTilingPass(),
                           enablePass);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createLinalgFillToLinalgGenericPass(),
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
  addPotentiallyNestedPass(pm, createFHEMaxTransformPass(), enablePass);
  addPotentiallyNestedPass(pm, createDynamicTLUPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerFHELinalgToLinalg(mlir::MLIRContext &context, mlir::ModuleOp &module,
                       std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("FHELinalgToLinalg", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertFHETensorOpsToLinalg(), enablePass);
  addPotentiallyNestedPass(pm, mlir::createLinalgGeneralizationPass(),
                           enablePass);
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerLinalgToLoops(mlir::MLIRContext &context, mlir::ModuleOp &module,
                   std::function<bool(mlir::Pass *)> enablePass,
                   bool parallelizeLoops) {
  mlir::PassManager pm(&context);
  pipelinePrinting("LinalgToLoops", pm, context);

  addPotentiallyNestedPass(
      pm,
      mlir::concretelang::createLinalgGenericOpWithTensorsToLoopsPass(
          parallelizeLoops),
      enablePass);

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
transformFHEBigInt(mlir::MLIRContext &context, mlir::ModuleOp &module,
                   std::function<bool(mlir::Pass *)> enablePass,
                   unsigned int chunkSize, unsigned int chunkWidth) {
  mlir::PassManager pm(&context);
  addPotentiallyNestedPass(
      pm,
      mlir::concretelang::createFHEBigIntTransformPass(chunkSize, chunkWidth),
      enablePass);
  // We want to fully unroll for loops introduced by the BigInt transform since
  // MANP doesn't support loops. This is a workaround that make the IR much
  // bigger than it should be
  addPotentiallyNestedPass(pm, mlir::createLoopUnrollPass(-1, false, true),
                           enablePass);
  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerFHEToTFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
               std::optional<V0FHEContext> &fheContext,
               std::function<bool(mlir::Pass *)> enablePass) {
  if (!fheContext)
    return mlir::success();

  mlir::PassManager pm(&context);
  auto solution = fheContext.value().solution;
  auto optCrt = getCrtDecompositionFromSolution(solution);
  if (optCrt) {
    pipelinePrinting("FHEToTFHECrt", pm, context);
    auto mods = mlir::SmallVector<int64_t>(optCrt->begin(), optCrt->end());
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createConvertFHEToTFHECrtPass(
            mlir::concretelang::CrtLoweringParameters(mods)),
        enablePass);
  } else {
    pipelinePrinting("FHEToTFHEScalar", pm, context);
    size_t polySize = getPolynomialSizeFromSolution(solution);
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createConvertFHEToTFHEScalarPass(
            mlir::concretelang::ScalarLoweringParameters(polySize)),
        enablePass);
  }

  return pm.run(module.getOperation());
}

mlir::LogicalResult
parametrizeTFHE(mlir::MLIRContext &context, mlir::ModuleOp &module,
                std::optional<V0FHEContext> &fheContext,
                std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("ParametrizeTFHE", pm, context);

  if (!fheContext) {
    // For tests, which invoke the pipeline without determining FHE
    // parameters
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createTFHECircuitSolutionParametrizationPass(
            std::nullopt),
        enablePass);
  } else if (auto monoSolution =
                 std::get_if<V0Parameter>(&fheContext->solution);
             monoSolution != nullptr) {
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createConvertTFHEGlobalParametrizationPass(
            *monoSolution),
        enablePass);
  } else if (auto circuitSolution =
                 std::get_if<optimizer::CircuitSolution>(&fheContext->solution);
             circuitSolution != nullptr) {
    addPotentiallyNestedPass(
        pm,
        mlir::concretelang::createTFHECircuitSolutionParametrizationPass(
            *circuitSolution),
        enablePass);
  }

  addPotentiallyNestedPass(pm, mlir::createCanonicalizerPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult batchTFHE(mlir::MLIRContext &context,
                              mlir::ModuleOp &module,
                              std::function<bool(mlir::Pass *)> enablePass,
                              int64_t maxBatchSize) {
  mlir::PassManager pm(&context);
  pipelinePrinting("BatchTFHE", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createBatchingPass(maxBatchSize), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
normalizeTFHEKeys(mlir::MLIRContext &context, mlir::ModuleOp &module,
                  std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TFHEKeyNormalization", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createTFHEKeyNormalizationPass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
extractTFHEStatistics(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass,
                      CompilationFeedback &feedback) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TFHEStatistics", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createStatisticExtractionPass(feedback),
      enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
lowerTFHEToConcrete(mlir::MLIRContext &context, mlir::ModuleOp &module,
                    std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TFHEToConcrete", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertTFHEToConcretePass(), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult
computeMemoryUsage(mlir::MLIRContext &context, mlir::ModuleOp &module,
                   std::function<bool(mlir::Pass *)> enablePass,
                   CompilationFeedback &feedback) {
  mlir::PassManager pm(&context);
  pipelinePrinting("Computing Memory Usage", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createMemoryUsagePass(feedback), enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult optimizeTFHE(mlir::MLIRContext &context,
                                 mlir::ModuleOp &module,
                                 std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TFHEOptimization", pm, context);
  addPotentiallyNestedPass(pm, mlir::concretelang::createTFHEOptimizationPass(),
                           enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult simulateTFHE(mlir::MLIRContext &context,
                                 mlir::ModuleOp &module,
                                 std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("TFHESimulation", pm, context);
  addPotentiallyNestedPass(pm, mlir::concretelang::createSimulateTFHEPass(),
                           enablePass);

  return pm.run(module.getOperation());
}

mlir::LogicalResult extractSDFGOps(mlir::MLIRContext &context,
                                   mlir::ModuleOp &module,
                                   std::function<bool(mlir::Pass *)> enablePass,
                                   bool unroll) {
  mlir::PassManager pm(&context);
  pipelinePrinting("extract SDFG ops from Concrete", pm, context);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createExtractSDFGOpsPass(unroll), enablePass);
  LogicalResult res = pm.run(module.getOperation());

  return res;
}

mlir::LogicalResult
addRuntimeContext(mlir::MLIRContext &context, mlir::ModuleOp &module,
                  std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("Adding Runtime Context", pm, context);
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

mlir::LogicalResult lowerToStd(mlir::MLIRContext &context,
                               mlir::ModuleOp &module,
                               std::function<bool(mlir::Pass *)> enablePass,
                               bool parallelizeLoops) {
  mlir::PassManager pm(&context);
  pipelinePrinting("Lowering to Std", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createTensorEmptyToBufferizationAllocPass(),
      enablePass);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createSCFForallToSCFForPass(), enablePass);

  // Bufferize
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.allowReturnAllocs = true;
  bufferizationOptions.printConflicts = true;
  bufferizationOptions.unknownTypeConverterFn =
      [](Value value, Attribute memorySpace,
         const mlir::bufferization::BufferizationOptions &options) {
        return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
            value.getType().cast<TensorType>(), memorySpace);
      };
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  bufferizationOptions.createDeallocs = false;

  std::unique_ptr<mlir::Pass> comprBuffPass =
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions);

  addPotentiallyNestedPass(pm, std::move(comprBuffPass), enablePass);

  // The bufferization may create `linalg.map` operations; Add another
  // conversion pass from linalg to loops
  addPotentiallyNestedPass(pm, mlir::createConvertLinalgToLoopsPass(),
                           enablePass);

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

  return pm.run(module);
}

mlir::LogicalResult lowerToCAPI(mlir::MLIRContext &context,
                                mlir::ModuleOp &module,
                                std::function<bool(mlir::Pass *)> enablePass,
                                bool gpu) {
  mlir::PassManager pm(&context);
  pipelinePrinting("Lowering to CAPI", pm, context);

  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertConcreteToCAPIPass(gpu), enablePass);
  addPotentiallyNestedPass(
      pm, mlir::concretelang::createConvertTracingToCAPIPass(), enablePass);

  return pm.run(module);
}

mlir::LogicalResult
lowerStdToLLVMDialect(mlir::MLIRContext &context, mlir::ModuleOp &module,
                      std::function<bool(mlir::Pass *)> enablePass) {
  mlir::PassManager pm(&context);
  pipelinePrinting("StdToLLVM", pm, context);

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
  // -O3 is done LLVMEmitFile.cpp
  auto optLevel = llvm::CodeGenOpt::None;
  std::function<llvm::Error(llvm::Module *)> optPipeline =
      mlir::makeOptimizingTransformer(optLevel, 0, nullptr);

  if (optPipeline(&module))
    return mlir::failure();
  else
    return mlir::success();
}

} // namespace pipeline
} // namespace concretelang
} // namespace mlir
