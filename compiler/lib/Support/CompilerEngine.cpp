// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <fstream>
#include <iostream>
#include <mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <stdio.h>
#include <string>

#include <llvm/Support/Error.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser/Parser.h>

#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include <concretelang/ClientLib/ClientParameters.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteDialect.h>
#include <concretelang/Dialect/Concrete/Transforms/BufferizableOpInterfaceImpl.h>
#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/Transforms/BufferizableOpInterfaceImpl.h>
#include <concretelang/Dialect/SDFG/IR/SDFGDialect.h>
#include <concretelang/Dialect/SDFG/Transforms/BufferizableOpInterfaceImpl.h>
#include <concretelang/Dialect/SDFG/Transforms/SDFGConvertibleOpInterfaceImpl.h>
#include <concretelang/Dialect/TFHE/IR/TFHEDialect.h>
#include <concretelang/Dialect/Tracing/IR/TracingDialect.h>
#include <concretelang/Dialect/Tracing/Transforms/BufferizableOpInterfaceImpl.h>
#include <concretelang/Runtime/DFRuntime.hpp>
#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/LLVMEmitFile.h>
#include <concretelang/Support/Pipeline.h>

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
    RT::registerBufferizableOpInterfaceExternalModels(registry);
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

void CompilerEngine::setGenerateClientParameters(bool v) {
  this->generateClientParameters = v;
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
llvm::Expected<llvm::Optional<optimizer::Description>>
CompilerEngine::getConcreteOptimizerDescription(CompilationResult &res) {
  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();
  mlir::ModuleOp module = res.mlirModuleRef->get();
  // If the values has been overwritten returns
  if (this->overrideMaxEintPrecision.hasValue() &&
      this->overrideMaxMANP.hasValue()) {
    auto constraint = mlir::concretelang::V0FHEConstraint{
        this->overrideMaxMANP.getValue(),
        this->overrideMaxEintPrecision.getValue()};
    return optimizer::Description{constraint, llvm::None};
  }
  auto config = this->compilerOptions.optimizerConfig;
  auto descriptions = mlir::concretelang::pipeline::getFHEContextFromFHE(
      mlirContext, module, config, enablePass);
  if (auto err = descriptions.takeError()) {
    return std::move(err);
  }
  if (descriptions->empty()) { // The pass has not been run
    return llvm::None;
  }
  if (this->compilerOptions.clientParametersFuncName.hasValue()) {
    auto name = this->compilerOptions.clientParametersFuncName.getValue();
    auto description = descriptions->find(name);
    if (description == descriptions->end()) {
      std::string names;
      for (auto &entry : *descriptions) {
        names += "'" + entry.first + "' ";
      }
      return StreamStringError()
             << "Could not find existing crypto parameters for function '"
             << name << "' (known functions: " << names << ")";
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
  if (compilerOptions.v0Parameter.hasValue()) {
    // parameters come from the compiler options
    auto v0Params = compilerOptions.v0Parameter.value();
    if (compilerOptions.largeIntegerParameter.hasValue()) {
      v0Params.largeInteger = compilerOptions.largeIntegerParameter;
    }
    V0FHEConstraint constraint;
    if (compilerOptions.v0FHEConstraints.hasValue()) {
      constraint = compilerOptions.v0FHEConstraints.value();
    }
    res.fheContext.emplace(
        mlir::concretelang::V0FHEContext{constraint, v0Params});
    return llvm::Error::success();
  }
  // compute parameters
  else {
    auto descr = getConcreteOptimizerDescription(res);
    if (auto err = descr.takeError()) {
      return err;
    }
    if (!descr.get().hasValue()) {
      return llvm::Error::success();
    }
    CompilationFeedback feedback;
    // Make sure to use the gpu constraint of the optimizer if we use gpu
    // backend.
    compilerOptions.optimizerConfig.use_gpu_constraints =
        compilerOptions.emitGPUOps;
    auto v0Params = getParameter(descr.get().value(), feedback,
                                 compilerOptions.optimizerConfig);
    if (auto err = v0Params.takeError()) {
      return err;
    }
    res.fheContext.emplace(mlir::concretelang::V0FHEContext{
        descr.get().value().constraint, v0Params.get()});
    res.feedback.emplace(feedback);
  }

  return llvm::Error::success();
}

using OptionalLib = llvm::Optional<std::shared_ptr<CompilerEngine::Library>>;
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

  // enable/disable usage of gpu functions during bufferization
  EMIT_GPU_OPS = options.emitGPUOps;

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

  auto dataflowParallelize =
      options.autoParallelize || options.dataflowParallelize;
  auto loopParallelize = options.autoParallelize || options.loopParallelize;
  if (options.verifyDiagnostics) {
    if (smHandler->verify().failed())
      return StreamStringError("Verification of diagnostics failed");
    else
      return std::move(res);
  }

  if (loopParallelize)
    mlir::concretelang::dfr::_dfr_set_use_omp(true);

  if (dataflowParallelize)
    mlir::concretelang::dfr::_dfr_set_required(true);

  if (!mlirModuleRef) {
    return errorDiag("Could not parse source");
  }

  res.mlirModuleRef = std::move(mlirModuleRef);
  mlir::ModuleOp module = res.mlirModuleRef->get();

  if (target == Target::ROUND_TRIP)
    return std::move(res);

  if (mlir::concretelang::pipeline::transformFHEBoolean(mlirContext, module,
                                                        enablePass)
          .failed()) {
    return errorDiag("Transforming FHE boolean ops failed");
  }

  if (options.chunkIntegers) {
    if (mlir::concretelang::pipeline::transformFHEBigInt(
            mlirContext, module, enablePass, options.chunkSize,
            options.chunkWidth)
            .failed()) {
      return errorDiag("Transforming FHE big integer ops failed");
    }
  }

  // FHE High level pass to determine FHE parameters
  if (auto err = this->determineFHEParameters(res))
    return std::move(err);

  // FHELinalg tiling
  if (options.fhelinalgTileSizes) {
    if (mlir::concretelang::pipeline::markFHELinalgForTiling(
            mlirContext, module, *options.fhelinalgTileSizes, enablePass)
            .failed())
      return errorDiag("Marking of FHELinalg operations for tiling failed");
  }

  if (mlir::concretelang::pipeline::tileMarkedFHELinalg(mlirContext, module,
                                                        enablePass)
          .failed()) {
    return errorDiag("Tiling of FHELinalg operations failed");
  }

  // Dataflow parallelization
  if (dataflowParallelize &&
      mlir::concretelang::pipeline::autopar(mlirContext, module, enablePass)
          .failed()) {
    return StreamStringError("Dataflow parallelization failed");
  }

  if (target == Target::FHE)
    return std::move(res);

  // FHELinalg -> FHE
  if (mlir::concretelang::pipeline::lowerFHELinalgToFHE(
          mlirContext, module, res.fheContext, enablePass, loopParallelize,
          options.batchConcreteOps)
          .failed()) {
    return errorDiag("Lowering from FHELinalg to FHE failed");
  }

  if (mlir::concretelang::pipeline::transformHighLevelFHEOps(mlirContext,
                                                             module, enablePass)
          .failed()) {
    return StreamStringError("Rewriting of high level fhe ops failed");
  }

  if (target == Target::FHE_NO_LINALG)
    return std::move(res);

  // Generate client parameters if requested
  if (this->generateClientParameters) {
    if (!options.clientParametersFuncName.hasValue()) {
      return StreamStringError(
          "Generation of client parameters requested, but no function name "
          "specified");
    }
    if (!res.fheContext.hasValue()) {
      return StreamStringError(
          "Cannot generate client parameters, the fhe context is empty for " +
          options.clientParametersFuncName.getValue());
    }
  }
  // Generate client parameters if requested
  auto funcName = options.clientParametersFuncName.getValueOr("main");
  if (this->generateClientParameters || target == Target::LIBRARY) {
    if (!res.fheContext.hasValue()) {
      // Some tests involve call a to non encrypted functions
      ClientParameters emptyParams;
      emptyParams.functionName = funcName;
      res.clientParameters = emptyParams;
    } else {
      llvm::Optional<::concretelang::clientlib::ChunkInfo> chunkInfo =
          llvm::None;
      if (options.chunkIntegers) {
        chunkInfo = ::concretelang::clientlib::ChunkInfo{4, 2};
      }
      auto clientParametersOrErr =
          mlir::concretelang::createClientParametersForV0(
              *res.fheContext, funcName, module,
              options.optimizerConfig.security, chunkInfo);
      if (!clientParametersOrErr)
        return clientParametersOrErr.takeError();

      res.clientParameters = clientParametersOrErr.get();
      res.feedback->fillFromClientParameters(*res.clientParameters);
    }
  }

  // FHE -> TFHE
  if (mlir::concretelang::pipeline::lowerFHEToTFHE(mlirContext, module,
                                                   res.fheContext, enablePass)
          .failed()) {
    return errorDiag("Lowering from FHE to TFHE failed");
  }

  // Optimizing TFHE
  if (this->compilerOptions.optimizeTFHE &&
      mlir::concretelang::pipeline::optimizeTFHE(mlirContext, module,
                                                 this->enablePass)
          .failed()) {
    return errorDiag("Optimizing TFHE failed");
  }

  if (target == Target::TFHE)
    return std::move(res);

  // TFHE -> Concrete
  if (mlir::concretelang::pipeline::lowerTFHEToConcrete(
          mlirContext, module, res.fheContext, this->enablePass)
          .failed()) {
    return errorDiag("Lowering from TFHE to Concrete failed");
  }

  if (target == Target::CONCRETE)
    return std::move(res);

  // Extract SDFG data flow graph from Concrete representation

  if (options.emitSDFGOps) {
    if (mlir::concretelang::pipeline::extractSDFGOps(
            mlirContext, module, enablePass,
            options.unrollLoopsWithSDFGConvertibleOps)
            .failed()) {
      return errorDiag("Extraction of SDFG operations from Concrete "
                       "representation failed");
    }
  }

  if (target == Target::SDFG) {
    return std::move(res);
  }

  // Concrete -> Canonical dialects
  if (mlir::concretelang::pipeline::lowerConcreteToStd(mlirContext, module,
                                                       enablePass)
          .failed()) {
    return errorDiag("Lowering from Bufferized Concrete to canonical MLIR "
                     "dialects failed");
  }

  // SDFG -> Canonical dialects
  if (mlir::concretelang::pipeline::lowerSDFGToStd(mlirContext, module,
                                                   enablePass)
          .failed()) {
    return errorDiag("Lowering from SDFG to canonical MLIR dialects failed");
  }

  if (target == Target::STD)
    return std::move(res);

  // MLIR canonical dialects -> LLVM Dialect
  if (mlir::concretelang::pipeline::lowerStdToLLVMDialect(
          mlirContext, module, enablePass, loopParallelize, options.emitGPUOps)
          .failed()) {
    return errorDiag("Failed to lower to LLVM dialect");
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
    return errorDiag("Failed to optimize LLVM IR");
  }

  if (target == Target::OPTIMIZED_LLVM_IR)
    return std::move(res);

  if (target == Target::LIBRARY) {
    if (!lib) {
      return StreamStringError(
          "Internal Error: Please provide a library parameter");
    }
    auto objPath = lib.getValue()->addCompilation(res);
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

llvm::Expected<CompilerEngine::Library> CompilerEngine::compile(
    std::vector<std::string> inputs, std::string outputDirPath,
    std::string runtimeLibraryPath, bool generateSharedLib,
    bool generateStaticLib, bool generateClientParameters,
    bool generateCompilationFeedback, bool generateCppHeader) {
  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(outputDirPath, runtimeLibraryPath);
  auto target = CompilerEngine::Target::LIBRARY;
  for (auto input : inputs) {
    auto compilation = compile(input, target, outputLib);
    if (!compilation) {
      return StreamStringError("Can't compile: ")
             << llvm::toString(compilation.takeError());
    }
  }
  if (auto err = outputLib->emitArtifacts(
          generateSharedLib, generateStaticLib, generateClientParameters,
          generateCompilationFeedback, generateCppHeader)) {
    return StreamStringError("Can't emit artifacts: ")
           << llvm::toString(std::move(err));
  }
  return *outputLib.get();
}

llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(llvm::SourceMgr &sm, std::string outputDirPath,
                        std::string runtimeLibraryPath, bool generateSharedLib,
                        bool generateStaticLib, bool generateClientParameters,
                        bool generateCompilationFeedback,
                        bool generateCppHeader) {
  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(outputDirPath, runtimeLibraryPath);
  auto target = CompilerEngine::Target::LIBRARY;

  auto compilation = compile(sm, target, outputLib);
  if (!compilation) {
    return StreamStringError("Can't compile: ")
           << llvm::toString(compilation.takeError());
  }

  if (auto err = outputLib->emitArtifacts(
          generateSharedLib, generateStaticLib, generateClientParameters,
          generateCompilationFeedback, generateCppHeader)) {
    return StreamStringError("Can't emit artifacts: ")
           << llvm::toString(std::move(err));
  }
  return *outputLib.get();
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
CompilerEngine::Library::getClientParametersPath(std::string outputDirPath) {
  llvm::SmallString<0> clientParametersPath(outputDirPath);
  llvm::sys::path::append(
      clientParametersPath,
      ClientParameters::getClientParametersPath("client_parameters"));
  return clientParametersPath.str().str();
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
// linking, this symbols should be available during runtime however. This is
// the case when JIT compiling, the JIT should either link to the runtime
// library that has the missing symbols, or it would have been loaded even
// prior to that. Starting from Mac 11 (Big Sur), it appears we need to add -L
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

llvm::Expected<std::string>
CompilerEngine::Library::emitClientParametersJSON() {
  auto clientParamsPath = getClientParametersPath(outputDirPath);
  llvm::json::Value value(clientParametersList);
  std::error_code error;
  llvm::raw_fd_ostream out(clientParamsPath, error);

  if (error) {
    return StreamStringError("cannot emit client parameters, error: ")
           << error.message();
  }
  out << llvm::formatv("{0:2}", value);
  out.close();

  return clientParamsPath;
}

llvm::Expected<std::string>
CompilerEngine::Library::emitCompilationFeedbackJSON() {
  auto path = getCompilationFeedbackPath(outputDirPath);
  if (compilationFeedbackList.size() != 1) {
    return StreamStringError("multiple compilation feedback not supported");
  }
  llvm::json::Value value(compilationFeedbackList[0]);
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

static std::string ccpResultType(size_t rank) {
  if (rank == 0) {
    return "scalar_out";
  } else {
    return "tensor" + std::to_string(rank) + "_out";
  }
}

static std::string ccpArgType(size_t rank) {
  if (rank == 0) {
    return "scalar_in";
  } else {
    return "tensor" + std::to_string(rank) + "_in";
  }
}

static std::string cppArgsType(std::vector<CircuitGate> inputs) {
  std::string args;
  for (auto input : inputs) {
    if (!args.empty()) {
      args += ", ";
    }
    args += ccpArgType(input.shape.dimensions.size());
  }
  return args;
}

llvm::Expected<std::string> CompilerEngine::Library::emitCppHeader() {
  std::string libraryName = "fhecircuit";
  auto headerName = libraryName + "-client.h";
  llvm::SmallString<0> headerPath(outputDirPath);
  llvm::sys::path::append(headerPath, headerName);

  std::error_code error;
  llvm::raw_fd_ostream out(headerPath, error);
  if (error) {
    StreamStringError("Cannot emit header: ")
        << headerPath << ", " << error.message() << "\n";
  }

  out << "#include \"boost/outcome.h\"\n";
  out << "#include \"concretelang/ClientLib/ClientLambda.h\"\n";
  out << "#include \"concretelang/ClientLib/KeySetCache.h\"\n";
  out << "#include \"concretelang/ClientLib/Types.h\"\n";
  out << "#include \"concretelang/Common/Error.h\"\n";
  out << "\n";
  out << "namespace " << libraryName << " {\n";
  out << "namespace client {\n";

  for (auto params : clientParametersList) {
    std::string args;
    std::string result;
    if (params.outputs.size() > 0) {
      args = cppArgsType(params.inputs);
    } else {
      args = "void";
    }
    if (params.outputs.size() > 0) {
      size_t rank = params.outputs[0].shape.dimensions.size();
      result = ccpResultType(rank);
    } else {
      result = "void";
    }
    out << "\n";
    out << "namespace " << params.functionName << " {\n";
    out << "  using namespace concretelang::clientlib;\n";
    out << "  using concretelang::error::StringError;\n";
    out << "  using " << params.functionName << "_t = TypedClientLambda<"
        << result << ", " << args << ">;\n";
    out << "  static const std::string name = \"" << params.functionName
        << "\";\n";
    out << "\n";
    out << "  static outcome::checked<" << params.functionName
        << "_t, StringError>\n";
    out << "  load(std::string outputLib)\n";
    out << "  { return " << params.functionName
        << "_t::load(name, outputLib); }\n";
    out << "} // namespace " << params.functionName << "\n";
  }
  out << "\n";
  out << "} // namespace client\n";
  out << "} // namespace " << libraryName << "\n";

  out.close();

  return headerPath.str().str();
}

llvm::Expected<std::string>
CompilerEngine::Library::addCompilation(CompilationResult &compilation) {
  llvm::Module *module = compilation.llvmModule.get();
  auto sourceName = module->getSourceFileName();
  if (sourceName == "" || sourceName == "LLVMDialectModule") {
    sourceName = this->outputDirPath + ".module-" +
                 std::to_string(objectsPath.size()) + ".mlir";
  }
  auto objectPath = sourceName + OBJECT_EXT;
  if (auto error = mlir::concretelang::emitObject(*module, objectPath)) {
    return std::move(error);
  }

  addExtraObjectFilePath(objectPath);
  if (compilation.clientParameters.hasValue()) {
    clientParametersList.push_back(compilation.clientParameters.getValue());
  }
  if (compilation.feedback.hasValue()) {
    compilationFeedbackList.push_back(compilation.feedback.getValue());
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
    llvm::Optional<std::vector<std::string>> extraArgs) {
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
                                                   bool compilationFeedback,
                                                   bool cppHeader) {
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
    if (auto err = emitClientParametersJSON().takeError()) {
      return err;
    }
  }
  if (compilationFeedback) {
    if (auto err = emitCompilationFeedbackJSON().takeError()) {
      return err;
    }
  }
  if (cppHeader) {
    if (auto err = emitCppHeader().takeError()) {
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
