// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "llvm/Support/Debug.h"
#include <err.h>
#include <fstream>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <optional>
#include <stdio.h>
#include <string>

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"

#include "concrete-protocol.capnp.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/RT/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Dialect/SDFG/Transforms/SDFGConvertibleOpInterfaceImpl.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"
#include "concretelang/Dialect/Tracing/Transforms/BufferizableOpInterfaceImpl.h"
#include "concretelang/Dialect/TypeInference/IR/TypeInferenceDialect.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/LLVMEmitFile.h"
#include "concretelang/Support/Pipeline.h"
#include "concretelang/Support/Utils.h"

namespace mlir {
namespace concretelang {
// TODO: should be removed when bufferization is not related to CAPI lowering
// Control whether we should call a cpu of gpu function when lowering
// to CAPI
static bool EMIT_GPU_OPS;
bool getEmitGPUOption() { return EMIT_GPU_OPS; }

/// Creates a new compilation context that can be shared across
/// compilation engines and results
std::shared_ptr<CompilationContext> CompilationContext::createShared() {
  return std::make_shared<CompilationContext>();
}

CompilationContext::CompilationContext()
    : mlirContext(nullptr), llvmContext(nullptr) {}

CompilationContext::~CompilationContext() {
  delete this->mlirContext;
  delete this->llvmContext;
}

/// Returns the MLIR context for a compilation context. Creates and
/// initializes a new MLIR context if necessary.
mlir::MLIRContext *CompilationContext::getMLIRContext() {
  if (this->mlirContext == nullptr) {
    mlir::DialectRegistry registry;
    registry.insert<
        mlir::concretelang::TypeInference::TypeInferenceDialect,
        mlir::concretelang::Tracing::TracingDialect,
        mlir::concretelang::RT::RTDialect, mlir::concretelang::FHE::FHEDialect,
        mlir::concretelang::TFHE::TFHEDialect,
        mlir::concretelang::FHELinalg::FHELinalgDialect,
        mlir::concretelang::Concrete::ConcreteDialect,
        mlir::concretelang::SDFG::SDFGDialect, mlir::func::FuncDialect,
        mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect,
        mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
        mlir::omp::OpenMPDialect, mlir::bufferization::BufferizationDialect>();
    Tracing::registerBufferizableOpInterfaceExternalModels(registry);
    Concrete::registerBufferizableOpInterfaceExternalModels(registry);
    SDFG::registerSDFGConvertibleOpInterfaceExternalModels(registry);
    SDFG::registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    RT::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    this->mlirContext = new mlir::MLIRContext();
    this->mlirContext->appendDialectRegistry(registry);
    this->mlirContext->loadAllAvailableDialects();
    this->mlirContext->disableMultithreading();
  }

  return this->mlirContext;
}

/// Returns the LLVM context for a compilation context. Creates and
/// initializes a new LLVM context if necessary.
llvm::LLVMContext *CompilationContext::getLLVMContext() {
  if (this->llvmContext == nullptr)
    this->llvmContext = new llvm::LLVMContext();

  return this->llvmContext;
}

/// Sets the FHE constraints for the compilation. Overrides any
/// automatically detected configuration and prevents the autodetection
/// pass from running.
void CompilerEngine::setFHEConstraints(
    const mlir::concretelang::V0FHEConstraint &c) {
  this->overrideMaxEintPrecision = c.p;
  this->overrideMaxMANP = c.norm2;
}

void CompilerEngine::setGenerateProgramInfo(bool v) {
  this->generateProgramInfo = v;
}

void CompilerEngine::setMaxEintPrecision(size_t v) {
  this->overrideMaxEintPrecision = v;
}

void CompilerEngine::setMaxMANP(size_t v) { this->overrideMaxMANP = v; }

void CompilerEngine::setEnablePass(
    std::function<bool(mlir::Pass *)> enablePass) {
  this->enablePass = enablePass;
}

/// Returns the optimizer::Description
llvm::Expected<std::optional<optimizer::Description>>
CompilerEngine::getConcreteOptimizerDescription(CompilationResult &res) {
  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();
  mlir::ModuleOp module = res.mlirModuleRef->get();
  // If the values has been overwritten returns
  if (this->overrideMaxEintPrecision.has_value() &&
      this->overrideMaxMANP.has_value()) {
    auto constraint = mlir::concretelang::V0FHEConstraint{
        this->overrideMaxMANP.value(), this->overrideMaxEintPrecision.value()};
    return optimizer::Description{constraint, std::nullopt};
  }
  auto config = this->compilerOptions.optimizerConfig;
  auto descriptions = mlir::concretelang::pipeline::getFHEContextFromFHE(
      mlirContext, module, config, enablePass);
  if (auto err = descriptions.takeError()) {
    return std::move(err);
  }
  if (descriptions->empty()) { // The pass has not been run
    return std::nullopt;
  }
  if (this->compilerOptions.mainFuncName.has_value()) {
    auto name = this->compilerOptions.mainFuncName.value();
    auto description = descriptions->find(name);
    if (description == descriptions->end()) {
      std::string names;
      return StreamStringError("Function not found, name='")
             << name << "', cannot get optimizer description";
    }
    return std::move(description->second);
  }
  if (descriptions->size() != 1) {
    llvm::errs() << "Several crypto parameters exists: the function need to be "
                    "specified, taking the first one";
  }
  return std::move(descriptions->begin()->second);
}

/// set the fheContext field if the v0Constraint can be computed
/// set the fheContext field if the v0Constraint can be computed
llvm::Error CompilerEngine::determineFHEParameters(CompilationResult &res) {
  if (compilerOptions.v0Parameter.has_value()) {
    // parameters come from the compiler options
    auto v0Params = compilerOptions.v0Parameter.value();
    if (compilerOptions.largeIntegerParameter.has_value()) {
      v0Params.largeInteger = compilerOptions.largeIntegerParameter;
    }
    V0FHEConstraint constraint;
    if (compilerOptions.v0FHEConstraints.has_value()) {
      constraint = compilerOptions.v0FHEConstraints.value();
    }
    res.fheContext.emplace(
        mlir::concretelang::V0FHEContext{constraint, v0Params});

    CompilationFeedback feedback;
    res.feedback.emplace(feedback);

    return llvm::Error::success();
  }
  // compute parameters
  else {
    auto descr = getConcreteOptimizerDescription(res);
    if (auto err = descr.takeError()) {
      return err;
    }
    if (!descr.get().has_value()) {
      return llvm::Error::success();
    }
    CompilationFeedback feedback;
    // Make sure to use the gpu constraint of the optimizer if we use gpu
    // backend.
    compilerOptions.optimizerConfig.use_gpu_constraints =
        compilerOptions.emitGPUOps;
    auto expectedSolution = getSolution(descr.get().value(), feedback,
                                        compilerOptions.optimizerConfig);
    if (auto err = expectedSolution.takeError()) {
      return err;
    }
    res.fheContext.emplace(mlir::concretelang::V0FHEContext{
        descr.get().value().constraint, *expectedSolution});
    res.feedback.emplace(feedback);
  }

  return llvm::Error::success();
}

using OptionalLib = std::optional<std::shared_ptr<CompilerEngine::Library>>;
// Compile the sources managed by the source manager `sm` to the
// target dialect `target`. If successful, the result can be retrieved
// using `getModule()` and `getLLVMModule()`, respectively depending
// on the target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(llvm::SourceMgr &sm, Target target, OptionalLib lib) {
  std::unique_ptr<mlir::SourceMgrDiagnosticVerifierHandler> smHandler;
  std::string diagnosticsMsg;
  llvm::raw_string_ostream diagnosticsOS(diagnosticsMsg);
  auto errorDiag = [&](std::string prefixMsg)
      -> llvm::Expected<CompilerEngine::CompilationResult> {
    return StreamStringError(prefixMsg + "\n" + diagnosticsOS.str());
  };

  CompilationResult res(this->compilationContext);

  CompilationOptions &options = this->compilerOptions;

  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();

  if (options.verifyDiagnostics) {
    // Only build diagnostics verifier handler if diagnostics should
    // be verified in order to avoid diagnostic messages to be
    // consumed when they should appear on stderr.
    smHandler = std::make_unique<mlir::SourceMgrDiagnosticVerifierHandler>(
        sm, &mlirContext, diagnosticsOS);
  }

  mlirContext.printOpOnDiagnostic(false);

  mlir::OwningOpRef<mlir::ModuleOp> mlirModuleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sm, &mlirContext);

  if (options.verifyDiagnostics) {
    if (smHandler->verify().failed())
      return StreamStringError("Verification of diagnostics failed");
    else
      return std::move(res);
  }

  if (!mlirModuleRef) {
    return errorDiag("Could not parse source");
  }

  return compile(mlirModuleRef.release(), target, lib);
}

llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(mlir::ModuleOp moduleOp, Target target,
                        OptionalLib lib) {
  CompilationResult res(this->compilationContext);

  CompilationOptions &options = this->compilerOptions;

  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();

  // enable/disable usage of gpu functions during bufferization
  EMIT_GPU_OPS = options.emitGPUOps;

  auto dataflowParallelize =
      options.autoParallelize || options.dataflowParallelize;
  if (options.optimizerConfig.strategy == optimizer::Strategy::DAG_MULTI &&
      dataflowParallelize == true) {
    // FIXME: DF is not currently compatible with multi-parameters as
    // the generation of dataflow tasks obfuscates the code before the
    // analysis can be done. Until this is fixed we cannot allow both.
    dataflowParallelize = false;
    warnx("WARNING: dataflow parallelization is not compatible with the "
          "optimizer strategy [dag-multi]. Continuing with dataflow "
          "parallelization disabled.");
  }
  auto loopParallelize = options.autoParallelize || options.loopParallelize;

  if (loopParallelize)
    mlir::concretelang::dfr::_dfr_set_use_omp(true);

  if (dataflowParallelize)
    mlir::concretelang::dfr::_dfr_set_required(true);

  mlir::OwningOpRef<mlir::ModuleOp> mlirModuleRef(moduleOp);
  res.mlirModuleRef = std::move(mlirModuleRef);
  mlir::ModuleOp module = res.mlirModuleRef->get();

  if (target == Target::ROUND_TRIP)
    return std::move(res);

  // Retrieves the encoding informations before any transformation is performed
  // on the `FHE` dialect.
  if ((this->generateProgramInfo || target == Target::LIBRARY) &&
      !options.encodings) {
    auto funcName = options.mainFuncName.value_or("main");
    auto encodingInfosOrErr =
        mlir::concretelang::encodings::getCircuitEncodings(funcName, module);
    if (!encodingInfosOrErr) {
      return encodingInfosOrErr.takeError();
    }
    options.encodings = std::move(*encodingInfosOrErr);
  }

  if (mlir::concretelang::pipeline::transformFHEBoolean(mlirContext, module,
                                                        enablePass)
          .failed()) {
    return StreamStringError("Transforming FHE boolean ops failed");
  }

  if (options.chunkIntegers) {
    if (mlir::concretelang::pipeline::transformFHEBigInt(
            mlirContext, module, enablePass, options.chunkSize,
            options.chunkWidth)
            .failed()) {
      return StreamStringError("Transforming FHE big integer ops failed");
    }
  }

  // FHE High level pass to determine FHE parameters
  if (auto err = this->determineFHEParameters(res))
    return std::move(err);

  // Now that FHE Parameters were computed, we can set the encoding mode of
  // integer ciphered inputs.
  if ((this->generateProgramInfo || target == Target::LIBRARY)) {
    std::optional<
        Message<concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode>>
        maybeChunkInfo(std::nullopt);
    if (options.chunkIntegers) {
      auto chunkedMode = Message<
          concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode>();
      chunkedMode.asBuilder().setSize(options.chunkSize);
      chunkedMode.asBuilder().setWidth(options.chunkWidth);
      maybeChunkInfo = chunkedMode;
    }
    mlir::concretelang::encodings::setCircuitEncodingModes(
        *options.encodings, maybeChunkInfo, res.fheContext);
  }

  // FHELinalg tiling
  if (options.fhelinalgTileSizes) {
    if (mlir::concretelang::pipeline::markFHELinalgForTiling(
            mlirContext, module, *options.fhelinalgTileSizes, enablePass)
            .failed())
      return StreamStringError(
          "Marking of FHELinalg operations for tiling failed");
  }

  if (target == Target::FHE)
    return std::move(res);

  // FHELinalg -> FHE
  if (mlir::concretelang::pipeline::lowerFHELinalgToLinalg(mlirContext, module,
                                                           enablePass)
          .failed()) {
    return StreamStringError("Lowering from FHELinalg to Linalg failed");
  }

  if (target == Target::FHE_LINALG_GENERIC)
    return std::move(res);

  if (mlir::concretelang::pipeline::tileMarkedLinalg(mlirContext, module,
                                                     enablePass)
          .failed()) {
    return StreamStringError("Tiling of Linalg operations failed");
  }

  if (mlir::concretelang::pipeline::lowerLinalgToLoops(
          mlirContext, module, enablePass, loopParallelize)
          .failed()) {
    return StreamStringError("Lowering from Linalg to loops failed");
  }

  if (mlir::concretelang::pipeline::transformHighLevelFHEOps(mlirContext,
                                                             module, enablePass)
          .failed()) {
    return StreamStringError("Rewriting of high level fhe ops failed");
  }

  // TODO: bring determineFHEParameters call here after the FHELinalg -> FHE
  // lowering
  // require to first support linalg.genric in the Optimizer Dag creation
  // FHE High level pass to determine FHE parameters
  // if (auto err = this->determineFHEParameters(res))
  //   return std::move(err);

  if (target == Target::FHE_NO_LINALG)
    return std::move(res);

  // Dataflow parallelization
  if (dataflowParallelize &&
      mlir::concretelang::pipeline::autopar(mlirContext, module, enablePass)
          .failed()) {
    return StreamStringError("Dataflow parallelization failed");
  }

  if (mlir::concretelang::pipeline::lowerLinalgToLoops(
          mlirContext, module, enablePass, loopParallelize)
          .failed()) {
    return StreamStringError("Lowering from Linalg Generic to Loops failed");
  }

  // FHE -> TFHE
  if (mlir::concretelang::pipeline::lowerFHEToTFHE(mlirContext, module,
                                                   res.fheContext, enablePass)
          .failed()) {
    return StreamStringError("Lowering from FHE to TFHE failed");
  }

  // Optimizing TFHE
  if (this->compilerOptions.optimizeTFHE &&
      mlir::concretelang::pipeline::optimizeTFHE(mlirContext, module,
                                                 this->enablePass)
          .failed()) {
    return StreamStringError("Optimizing TFHE failed");
  }

  if (target == Target::TFHE)
    return std::move(res);

  if (mlir::concretelang::pipeline::parametrizeTFHE(mlirContext, module,
                                                    res.fheContext, enablePass)
          .failed()) {
    return StreamStringError("Parametrization of TFHE operations failed");
  }

  if (target == Target::PARAMETRIZED_TFHE)
    return std::move(res);

  // Normalize TFHE keys
  if (mlir::concretelang::pipeline::normalizeTFHEKeys(mlirContext, module,
                                                      this->enablePass)
          .failed()) {
    return StreamStringError("Normalizing TFHE keys failed");
  }

  // Generate client parameters if requested
  if (this->generateProgramInfo) {
    if (!options.mainFuncName.has_value()) {
      return StreamStringError(
          "Generation of client parameters requested, but no function name "
          "specified");
    }
    if (!res.fheContext.has_value()) {
      return StreamStringError(
          "Cannot generate client parameters, the fhe context is empty for " +
          options.mainFuncName.value());
    }
  }
  // Generate program info if requested
  if (this->generateProgramInfo || target == Target::LIBRARY) {
    auto funcName = options.mainFuncName.value_or("main");
    if (!res.fheContext.has_value()) {
      // Some tests involve call a to non encrypted functions
      auto programInfo = Message<concreteprotocol::ProgramInfo>();
      programInfo.asBuilder().initCircuits(1);
      programInfo.asBuilder().getCircuits()[0].setName(std::string(funcName));
      res.programInfo = programInfo;
    } else {
      auto programInfoOrErr =
          mlir::concretelang::createProgramInfoFromTfheDialect(
              module, funcName, options.optimizerConfig.security,
              options.encodings.value(), options.compressEvaluationKeys);

      if (!programInfoOrErr)
        return programInfoOrErr.takeError();

      res.programInfo = std::move(*programInfoOrErr);
      // If more than one circuit, feedback can not be generated for now ..
      if (res.programInfo->asReader().getCircuits().size() != 1) {
        return StreamStringError(
            "Cannot generate feedback for program with more than one circuit.");
      }
      res.feedback->fillFromProgramInfo(*res.programInfo);
    }
  }

  if (target == Target::NORMALIZED_TFHE)
    return std::move(res);

  if (res.feedback) {
    if (mlir::concretelang::pipeline::extractTFHEStatistics(
            mlirContext, module, this->enablePass, res.feedback.value())
            .failed()) {
      return StreamStringError("Extracting TFHE statistics failed");
    }
  }

  if (options.simulate) {
    if (mlir::concretelang::pipeline::simulateTFHE(mlirContext, module,
                                                   this->enablePass)
            .failed()) {
      return StreamStringError("Simulating TFHE failed");
    }
  }

  if (target == Target::SIMULATED_TFHE)
    return std::move(res);

  if (options.batchTFHEOps) {
    if (mlir::concretelang::pipeline::batchTFHE(mlirContext, module, enablePass,
                                                options.maxBatchSize)
            .failed()) {
      return StreamStringError("Batching of TFHE operations");
    }
  }

  if (target == Target::BATCHED_TFHE)
    return std::move(res);

  // TFHE -> Concrete
  if (mlir::concretelang::pipeline::lowerTFHEToConcrete(mlirContext, module,
                                                        this->enablePass)
          .failed()) {
    return StreamStringError("Lowering from TFHE to Concrete failed");
  }

  if (target == Target::CONCRETE)
    return std::move(res);

  // Extract SDFG data flow graph from Concrete representation

  if (options.emitSDFGOps) {
    if (mlir::concretelang::pipeline::extractSDFGOps(
            mlirContext, module, enablePass,
            options.unrollLoopsWithSDFGConvertibleOps)
            .failed()) {
      return StreamStringError("Extraction of SDFG operations from Concrete "
                               "representation failed");
    }
  }

  if (target == Target::SDFG) {
    return std::move(res);
  }

  // Add runtime context in Concrete
  if (mlir::concretelang::pipeline::addRuntimeContext(mlirContext, module,
                                                      enablePass)
          .failed()) {
    return StreamStringError("Adding Runtime Context failed");
  }

  // SDFG -> Canonical dialects
  if (mlir::concretelang::pipeline::lowerSDFGToStd(mlirContext, module,
                                                   enablePass)
          .failed()) {
    return StreamStringError(
        "Lowering from SDFG to canonical MLIR dialects failed");
  }

  // bufferize and related passes
  if (mlir::concretelang::pipeline::lowerToStd(mlirContext, module, enablePass,
                                               loopParallelize)
          .failed()) {
    return StreamStringError("Failed to lower to std");
  }

  if (target == Target::STD)
    return std::move(res);

  if (res.feedback) {
    if (mlir::concretelang::pipeline::computeMemoryUsage(
            mlirContext, module, this->enablePass, res.feedback.value())
            .failed()) {
      return StreamStringError("Computing memory usage failed");
    }
  }

  if (mlir::concretelang::pipeline::lowerToCAPI(mlirContext, module, enablePass,
                                                options.emitGPUOps)
          .failed()) {
    return StreamStringError("Failed to lower to CAPI");
  }

  // MLIR canonical dialects -> LLVM Dialect
  if (mlir::concretelang::pipeline::lowerStdToLLVMDialect(mlirContext, module,
                                                          enablePass)
          .failed()) {
    return StreamStringError("Failed to lower to LLVM dialect");
  }

  if (target == Target::LLVM)
    return std::move(res);

  // Lowering to actual LLVM IR (i.e., not the LLVM dialect)
  llvm::LLVMContext &llvmContext = *this->compilationContext->getLLVMContext();

  res.llvmModule = mlir::concretelang::pipeline::lowerLLVMDialectToLLVMIR(
      mlirContext, llvmContext, module);

  if (!res.llvmModule)
    return StreamStringError("Failed to convert from LLVM dialect to LLVM IR");

  if (target == Target::LLVM_IR)
    return std::move(res);

  if (mlir::concretelang::pipeline::optimizeLLVMModule(llvmContext,
                                                       *res.llvmModule)
          .failed()) {
    return StreamStringError("Failed to optimize LLVM IR");
  }

  if (target == Target::OPTIMIZED_LLVM_IR)
    return std::move(res);

  if (target == Target::LIBRARY) {
    if (!lib) {
      return StreamStringError(
          "Internal Error: Please provide a library parameter");
    }
    auto objPath = lib.value()->setCompilationResult(res);
    if (!objPath) {
      return StreamStringError(llvm::toString(objPath.takeError()));
    }
    return std::move(res);
  }

  return std::move(res);
}

/// Compile the source `s` to the target dialect `target`. If successful, the
/// result can be retrieved using `getModule()` and `getLLVMModule()`,
/// respectively depending on the target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(llvm::StringRef s, Target target, OptionalLib lib) {
  std::unique_ptr<llvm::MemoryBuffer> mb = llvm::MemoryBuffer::getMemBuffer(s);
  return this->compile(std::move(mb), target, lib);
}

/// Compile the contained in `buffer` to the target dialect
/// `target`. If successful, the result can be retrieved using
/// `getModule()` and `getLLVMModule()`, respectively depending on the
/// target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(std::unique_ptr<llvm::MemoryBuffer> buffer,
                        Target target, OptionalLib lib) {
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  return this->compile(sm, target, lib);
}

llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(std::vector<std::string> inputs,
                        std::string outputDirPath,
                        std::string runtimeLibraryPath, bool generateSharedLib,
                        bool generateStaticLib, bool generateClientParameters,
                        bool generateCompilationFeedback) {
  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(outputDirPath, runtimeLibraryPath);
  auto target = CompilerEngine::Target::LIBRARY;
  for (auto input : inputs) {
    auto compilation = compile(input, target, outputLib);
    if (!compilation) {
      return compilation.takeError();
    }
  }
  if (auto err = outputLib->emitArtifacts(generateSharedLib, generateStaticLib,
                                          generateClientParameters,
                                          generateCompilationFeedback)) {
    return StreamStringError("Can't emit artifacts: ")
           << llvm::toString(std::move(err));
  }
  return *outputLib.get();
}

template <typename T>
llvm::Expected<CompilerEngine::Library>
compileModuleOrSource(CompilerEngine *engine, T module,
                      std::string outputDirPath, std::string runtimeLibraryPath,
                      bool generateSharedLib, bool generateStaticLib,
                      bool generateClientParameters,
                      bool generateCompilationFeedback) {
  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(outputDirPath, runtimeLibraryPath);
  auto target = CompilerEngine::Target::LIBRARY;

  auto compilation = engine->compile(module, target, outputLib);
  if (!compilation) {
    return compilation.takeError();
  }

  if (auto err = outputLib->emitArtifacts(generateSharedLib, generateStaticLib,
                                          generateClientParameters,
                                          generateCompilationFeedback)) {
    return StreamStringError("Can't emit artifacts: ")
           << llvm::toString(std::move(err));
  }
  return *outputLib.get();
}

llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(llvm::SourceMgr &sm, std::string outputDirPath,
                        std::string runtimeLibraryPath, bool generateSharedLib,
                        bool generateStaticLib, bool generateClientParameters,
                        bool generateCompilationFeedback) {
  return compileModuleOrSource<llvm::SourceMgr &>(
      this, sm, outputDirPath, runtimeLibraryPath, generateSharedLib,
      generateStaticLib, generateClientParameters, generateCompilationFeedback);
}

llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(mlir::ModuleOp module, std::string outputDirPath,
                        std::string runtimeLibraryPath, bool generateSharedLib,
                        bool generateStaticLib, bool generateClientParameters,
                        bool generateCompilationFeedback) {
  return compileModuleOrSource<mlir::ModuleOp>(
      this, module, outputDirPath, runtimeLibraryPath, generateSharedLib,
      generateStaticLib, generateClientParameters, generateCompilationFeedback);
}

/// Returns the path of the shared library
std::string
CompilerEngine::Library::getSharedLibraryPath(std::string outputDirPath) {
  llvm::SmallString<0> sharedLibraryPath(outputDirPath);
  llvm::sys::path::append(sharedLibraryPath, "sharedlib" + DOT_SHARED_LIB_EXT);
  return sharedLibraryPath.str().str();
}

/// Returns the path of the static library
std::string
CompilerEngine::Library::getStaticLibraryPath(std::string outputDirPath) {
  llvm::SmallString<0> staticLibraryPath(outputDirPath);
  llvm::sys::path::append(staticLibraryPath, "staticlib" + DOT_STATIC_LIB_EXT);
  return staticLibraryPath.str().str();
}

/// Returns the path of the client parameter
std::string
CompilerEngine::Library::getProgramInfoPath(std::string outputDirPath) {
  llvm::SmallString<0> programInfoPath(outputDirPath);
  llvm::sys::path::append(programInfoPath, "program_info.concrete.params.json");
  return programInfoPath.str().str();
}

/// Returns the path of the compiler feedback
std::string
CompilerEngine::Library::getCompilationFeedbackPath(std::string outputDirPath) {
  llvm::SmallString<0> compilationFeedbackPath(outputDirPath);
  llvm::sys::path::append(compilationFeedbackPath, "compilation_feedback.json");
  return compilationFeedbackPath.str().str();
}

const std::string CompilerEngine::Library::OBJECT_EXT = ".o";
const std::string CompilerEngine::Library::LINKER = "ld";
#ifdef __APPLE__
// We need to tell the linker that some symbols will be missing during
// linking, this symbols should be available during runtime however.
// Starting from Mac 11 (Big Sur), it appears we need to add -L
// /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lSystem for
// the sharedlib to link properly.
const std::string CompilerEngine::Library::LINKER_SHARED_OPT =
    " -dylib -undefined dynamic_lookup -L "
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lSystem "
    "-o ";
const std::string CompilerEngine::Library::DOT_SHARED_LIB_EXT = ".dylib";
#else // Linux
const std::string CompilerEngine::Library::LINKER_SHARED_OPT = " --shared -o ";
const std::string CompilerEngine::Library::DOT_SHARED_LIB_EXT = ".so";
#endif
const std::string CompilerEngine::Library::AR = "ar";
const std::string CompilerEngine::Library::AR_STATIC_OPT = " rcs ";
const std::string CompilerEngine::Library::DOT_STATIC_LIB_EXT = ".a";

void CompilerEngine::Library::addExtraObjectFilePath(std::string path) {
  objectsPath.push_back(path);
}

Message<concreteprotocol::ProgramInfo>
CompilerEngine::Library::getProgramInfo() const {
  return programInfo;
}

const std::string &CompilerEngine::Library::getOutputDirPath() const {
  return outputDirPath;
}

llvm::Expected<std::string> CompilerEngine::Library::emitProgramInfoJSON() {
  auto programInfoPath = getProgramInfoPath(outputDirPath);
  std::error_code error;
  llvm::raw_fd_ostream out(programInfoPath, error);
  auto maybeJson = programInfo.writeJsonToString();
  if (maybeJson.has_failure()) {
    return StreamStringError(maybeJson.as_failure().error().mesg);
  }
  auto json = maybeJson.value();
  out << json;
  out.close();

  return programInfoPath;
}

llvm::Expected<std::string>
CompilerEngine::Library::emitCompilationFeedbackJSON() {
  auto path = getCompilationFeedbackPath(outputDirPath);
  llvm::json::Value value(compilationFeedback);
  std::error_code error;
  llvm::raw_fd_ostream out(path, error);

  if (error) {
    return StreamStringError("cannot emit client parameters, error: ")
           << error.message();
  }
  out << llvm::formatv("{0:2}", value);
  out.close();

  return path;
}

llvm::Expected<std::string>
CompilerEngine::Library::setCompilationResult(CompilationResult &compilation) {
  llvm::Module *module = compilation.llvmModule.get();
  auto sourceName = module->getSourceFileName();
  if (sourceName == "" || sourceName == "LLVMDialectModule") {
    sourceName = this->outputDirPath + "/program.module-" +
                 std::to_string(objectsPath.size()) + ".mlir";
  }
  auto objectPath = sourceName + OBJECT_EXT;
  if (auto error = mlir::concretelang::emitObject(*module, objectPath)) {
    return std::move(error);
  }

  addExtraObjectFilePath(objectPath);
  if (compilation.programInfo) {
    programInfo = *compilation.programInfo;
  }
  if (compilation.feedback.has_value()) {
    compilationFeedback = compilation.feedback.value();
  }
  return objectPath;
}

bool stringEndsWith(std::string path, std::string requiredExt) {
  return path.substr(path.size() - requiredExt.size()) == requiredExt;
}

std::string removeDotExt(std::string path, std::string dotExt) {
  return (stringEndsWith(path, dotExt))
             ? path.substr(0, path.size() - dotExt.size())
             : path;
}

std::string ensureLibDotExt(std::string path, std::string dotExt) {
  path = removeDotExt(path, CompilerEngine::Library::DOT_STATIC_LIB_EXT);
  path = removeDotExt(path, CompilerEngine::Library::DOT_SHARED_LIB_EXT);
  return path + dotExt;
}

llvm::Expected<std::string> CompilerEngine::Library::emit(
    std::string path, std::string dotExt, std::string linker,
    std::optional<std::vector<std::string>> extraArgs) {
  auto pathDotExt = ensureLibDotExt(path, dotExt);
  auto error = mlir::concretelang::emitLibrary(objectsPath, pathDotExt, linker,
                                               extraArgs);
  if (error) {
    return std::move(error);
  }
  return pathDotExt;
}

llvm::Expected<std::string> CompilerEngine::Library::emitShared() {
  std::vector<std::string> extraArgs;
  std::string fullRuntimeLibraryName = "";
#ifdef __APPLE__
  // to issue the command for fixing the runtime dependency of the generated
  // lib
  bool fixRuntimeDep = false;
#endif
  if (!runtimeLibraryPath.empty()) {
    // Getting the parent dir should work on Linux and Mac
    std::size_t rpathLastPos = runtimeLibraryPath.find_last_of("/");
    std::string rpath = "";
    std::string runtimeLibraryName = "";
    if (rpathLastPos != std::string::npos) {
      rpath = runtimeLibraryPath.substr(0, rpathLastPos);
      fullRuntimeLibraryName = runtimeLibraryPath.substr(
          rpathLastPos + 1, runtimeLibraryPath.length());
      // runtimeLibraryName is part of fullRuntimeLibraryName =
      // lib(runtimeLibraryName).dylib
      runtimeLibraryName =
          removeDotExt(fullRuntimeLibraryName, DOT_SHARED_LIB_EXT);
      if (runtimeLibraryName.rfind("lib", 0) == 0) { // starts with lib
        runtimeLibraryName =
            runtimeLibraryName.substr(3, runtimeLibraryName.length());
      }
    }
#ifdef __APPLE__
    if (!rpath.empty() && !runtimeLibraryName.empty()) {
      fixRuntimeDep = true;
      extraArgs.push_back("-l" + runtimeLibraryName);
      extraArgs.push_back("-L" + rpath);
      extraArgs.push_back("-rpath " + rpath);
    }
#else // Linux
    extraArgs.push_back(runtimeLibraryPath);
    if (!rpath.empty()) {
      extraArgs.push_back("-rpath=" + rpath);
      // Use RPATH instead of RUNPATH for transitive dependencies
      extraArgs.push_back("--disable-new-dtags");
    }
#endif
  }
  auto path = emit(getSharedLibraryPath(outputDirPath), DOT_SHARED_LIB_EXT,
                   LINKER + LINKER_SHARED_OPT, extraArgs);
  if (path) {
    sharedLibraryPath = path.get();
#ifdef __APPLE__
    // when dellocate is used to include dependencies in python wheels, the
    // runtime library will have an id that is prefixed with /DLC, and that
    // path doesn't exist. So when generated libraries won't be able to find
    // it during load time. To solve this, we change the dep in the generated
    // library to be relative to the rpath which should be set correctly
    // during linking. This shouldn't have an impact when
    // /DLC/concrete/.dylibs/* isn't a dependecy in the first place (when not
    // using python).
    if (fixRuntimeDep) {
      std::string fixRuntimeDepCmd = "install_name_tool -change "
                                     "/DLC/concrete/.dylibs/" +
                                     fullRuntimeLibraryName + " @rpath/" +
                                     fullRuntimeLibraryName + " " +
                                     sharedLibraryPath;
      auto error = mlir::concretelang::callCmd(fixRuntimeDepCmd);
      if (error) {
        return std::move(error);
      }
    }
#endif
  }

  return path;
}

llvm::Expected<std::string> CompilerEngine::Library::emitStatic() {
  auto path = emit(getStaticLibraryPath(outputDirPath), DOT_STATIC_LIB_EXT,
                   AR + AR_STATIC_OPT);
  if (path) {
    staticLibraryPath = path.get();
  }
  return path;
}

llvm::Error CompilerEngine::Library::emitArtifacts(bool sharedLib,
                                                   bool staticLib,
                                                   bool clientParameters,
                                                   bool compilationFeedback) {
  // Create output directory if doesn't exist
  llvm::sys::fs::create_directory(outputDirPath);
  if (sharedLib) {
    if (auto err = emitShared().takeError()) {
      return err;
    }
  }
  if (staticLib) {
    if (auto err = emitStatic().takeError()) {
      return err;
    }
  }
  if (clientParameters) {
    if (auto err = emitProgramInfoJSON().takeError()) {
      return err;
    }
  }
  if (compilationFeedback) {
    if (auto err = emitCompilationFeedbackJSON().takeError()) {
      return err;
    }
  }
  return llvm::Error::success();
}

CompilerEngine::Library::~Library() {
  if (cleanUp) {
    for (auto path : objectsPath) {
      remove(path.c_str());
    }
  }
}

} // namespace concretelang
} // namespace mlir
