// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <fstream>
#include <iostream>
#include <regex>
#include <stdio.h>
#include <string>

#include <llvm/Support/Error.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <concretelang/ClientLib/ClientParameters.h>
#include <concretelang/Dialect/BConcrete/IR/BConcreteDialect.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteDialect.h>
#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/TFHE/IR/TFHEDialect.h>
#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/LLVMEmitFile.h>
#include <concretelang/Support/Pipeline.h>

namespace mlir {
namespace concretelang {

// Creates a new compilation context that can be shared across
// compilation engines and results
std::shared_ptr<CompilationContext> CompilationContext::createShared() {
  return std::make_shared<CompilationContext>();
}

CompilationContext::CompilationContext()
    : mlirContext(nullptr), llvmContext(nullptr) {}

CompilationContext::~CompilationContext() {
  delete this->mlirContext;
  delete this->llvmContext;
}

// Returns the MLIR context for a compilation context. Creates and
// initializes a new MLIR context if necessary.
mlir::MLIRContext *CompilationContext::getMLIRContext() {
  if (this->mlirContext == nullptr) {
    this->mlirContext = new mlir::MLIRContext();

    this->mlirContext->getOrLoadDialect<mlir::concretelang::RT::RTDialect>();
    this->mlirContext->getOrLoadDialect<mlir::concretelang::FHE::FHEDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::concretelang::TFHE::TFHEDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::concretelang::FHELinalg::FHELinalgDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::concretelang::Concrete::ConcreteDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::concretelang::BConcrete::BConcreteDialect>();
    this->mlirContext->getOrLoadDialect<mlir::StandardOpsDialect>();
    this->mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();
    this->mlirContext->getOrLoadDialect<mlir::linalg::LinalgDialect>();
    this->mlirContext->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    this->mlirContext->getOrLoadDialect<mlir::scf::SCFDialect>();
    this->mlirContext->getOrLoadDialect<mlir::omp::OpenMPDialect>();
  }

  return this->mlirContext;
}

// Returns the LLVM context for a compilation context. Creates and
// initializes a new LLVM context if necessary.
llvm::LLVMContext *CompilationContext::getLLVMContext() {
  if (this->llvmContext == nullptr)
    this->llvmContext = new llvm::LLVMContext();

  return this->llvmContext;
}

// Sets the FHE constraints for the compilation. Overrides any
// automatically detected configuration and prevents the autodetection
// pass from running.
void CompilerEngine::setFHEConstraints(
    const mlir::concretelang::V0FHEConstraint &c) {
  this->overrideMaxEintPrecision = c.p;
  this->overrideMaxMANP = c.norm2;
}

void CompilerEngine::setVerifyDiagnostics(bool v) {
  this->verifyDiagnostics = v;
}

void CompilerEngine::setAutoParallelize(bool v) { this->autoParallelize = v; }

void CompilerEngine::setLoopParallelize(bool v) { this->loopParallelize = v; }

void CompilerEngine::setDataflowParallelize(bool v) {
  this->dataflowParallelize = v;
}

void CompilerEngine::setGenerateClientParameters(bool v) {
  this->generateClientParameters = v;
}

void CompilerEngine::setMaxEintPrecision(size_t v) {
  this->overrideMaxEintPrecision = v;
}

void CompilerEngine::setMaxMANP(size_t v) { this->overrideMaxMANP = v; }

void CompilerEngine::setClientParametersFuncName(const llvm::StringRef &name) {
  this->clientParametersFuncName = name.str();
}

void CompilerEngine::setFHELinalgTileSizes(llvm::ArrayRef<int64_t> sizes) {
  this->fhelinalgTileSizes = sizes.vec();
}

void CompilerEngine::setEnablePass(
    std::function<bool(mlir::Pass *)> enablePass) {
  this->enablePass = enablePass;
}

// Returns the overwritten V0FHEConstraint or try to compute them from FHE
llvm::Expected<llvm::Optional<mlir::concretelang::V0FHEConstraint>>
CompilerEngine::getV0FHEConstraint(CompilationResult &res) {
  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();
  mlir::ModuleOp module = res.mlirModuleRef->get();
  llvm::Optional<mlir::concretelang::V0FHEConstraint> fheConstraints;
  // If the values has been overwritten returns
  if (this->overrideMaxEintPrecision.hasValue() &&
      this->overrideMaxMANP.hasValue()) {
    return mlir::concretelang::V0FHEConstraint{
        this->overrideMaxMANP.getValue(),
        this->overrideMaxEintPrecision.getValue()};
  }
  // Else compute constraint from FHE
  llvm::Expected<llvm::Optional<mlir::concretelang::V0FHEConstraint>>
      fheConstraintsOrErr =
          mlir::concretelang::pipeline::getFHEConstraintsFromFHE(
              mlirContext, module, enablePass);

  if (auto err = fheConstraintsOrErr.takeError())
    return std::move(err);

  return fheConstraintsOrErr.get();
}

// set the fheContext field if the v0Constraint can be computed
llvm::Error CompilerEngine::determineFHEParameters(CompilationResult &res) {
  auto fheConstraintOrErr = getV0FHEConstraint(res);
  if (auto err = fheConstraintOrErr.takeError())
    return std::move(err);
  if (!fheConstraintOrErr.get().hasValue()) {
    return llvm::Error::success();
  }
  const mlir::concretelang::V0Parameter *fheParams =
      getV0Parameter(fheConstraintOrErr.get().getValue());

  if (!fheParams) {
    return StreamStringError()
           << "Could not determine V0 parameters for 2-norm of "
           << (*fheConstraintOrErr)->norm2 << " and p of "
           << (*fheConstraintOrErr)->p;
  }
  res.fheContext.emplace(mlir::concretelang::V0FHEContext{
      (*fheConstraintOrErr).getValue(), *fheParams});

  return llvm::Error::success();
}

using OptionalLib = llvm::Optional<std::shared_ptr<CompilerEngine::Library>>;
// Compile the sources managed by the source manager `sm` to the
// target dialect `target`. If successful, the result can be retrieved
// using `getModule()` and `getLLVMModule()`, respectively depending
// on the target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(llvm::SourceMgr &sm, Target target, OptionalLib lib) {
  std::string diagnosticsMsg;
  llvm::raw_string_ostream diagnosticsOS(diagnosticsMsg);
  auto errorDiag = [&](std::string prefixMsg)
      -> llvm::Expected<CompilerEngine::CompilationResult> {
    return StreamStringError(prefixMsg + "\n" + diagnosticsOS.str());
  };

  CompilationResult res(this->compilationContext);

  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();

  mlir::SourceMgrDiagnosticVerifierHandler smHandler(sm, &mlirContext,
                                                     diagnosticsOS);
  mlirContext.printOpOnDiagnostic(false);

  mlir::OwningModuleRef mlirModuleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sm, &mlirContext);

  if (this->verifyDiagnostics) {
    if (smHandler.verify().failed())
      return StreamStringError("Verification of diagnostics failed");
    else
      return std::move(res);
  }

  if (!mlirModuleRef) {
    return errorDiag("Could not parse source");
  }

  res.mlirModuleRef = std::move(mlirModuleRef);
  mlir::ModuleOp module = res.mlirModuleRef->get();

  if (target == Target::ROUND_TRIP)
    return std::move(res);

  // FHE High level pass to determine FHE parameters
  if (auto err = this->determineFHEParameters(res))
    return std::move(err);

  // FHELinalg tiling
  if (this->fhelinalgTileSizes) {
    if (mlir::concretelang::pipeline::markFHELinalgForTiling(
            mlirContext, module, *this->fhelinalgTileSizes, enablePass)
            .failed())
      return errorDiag("Marking of FHELinalg operations for tiling failed");
  }

  if (mlir::concretelang::pipeline::tileMarkedFHELinalg(mlirContext, module,
                                                        enablePass)
          .failed()) {
    return errorDiag("Tiling of FHELinalg operations failed");
  }

  // Dataflow parallelization
  if ((this->autoParallelize || this->dataflowParallelize) &&
      mlir::concretelang::pipeline::autopar(mlirContext, module, enablePass)
          .failed()) {
    return StreamStringError("Dataflow parallelization failed");
  }

  if (target == Target::FHE)
    return std::move(res);

  // FHE -> TFHE
  if (mlir::concretelang::pipeline::lowerFHEToTFHE(mlirContext, module,
                                                   enablePass)
          .failed()) {
    return errorDiag("Lowering from FHE to TFHE failed");
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

  // Generate client parameters if requested
  if (this->generateClientParameters) {
    if (!this->clientParametersFuncName.hasValue()) {
      return StreamStringError(
          "Generation of client parameters requested, but no function name "
          "specified");
    }
    if (!res.fheContext.hasValue()) {
      return StreamStringError(
          "Cannot generate client parameters, the fhe context is empty");
    }
  }
  // Generate client parameters if requested
  auto funcName = this->clientParametersFuncName.getValueOr("main");
  if (this->generateClientParameters || target == Target::LIBRARY) {
    if (!res.fheContext.hasValue()) {
      // Some tests involve call a to non encrypted functions
      ClientParameters emptyParams;
      emptyParams.functionName = funcName;
      res.clientParameters = emptyParams;
    } else {
      auto clientParametersOrErr =
          mlir::concretelang::createClientParametersForV0(*res.fheContext,
                                                          funcName, module);
      if (!clientParametersOrErr)
        return clientParametersOrErr.takeError();

      res.clientParameters = clientParametersOrErr.get();
    }
  }

  // Concrete -> BConcrete
  if (mlir::concretelang::pipeline::lowerConcreteToBConcrete(
          mlirContext, module, this->enablePass)
          .failed()) {
    return StreamStringError(
        "Lowering from Concrete to Bufferized Concrete failed");
  }

  if (target == Target::BCONCRETE) {
    return std::move(res);
  }

  // BConcrete -> Canonical dialects
  if (mlir::concretelang::pipeline::lowerBConcreteToStd(mlirContext, module,
                                                        enablePass)
          .failed()) {
    return errorDiag(
        "Lowering from Bufferized Concrete to canonical MLIR dialects failed");
  }
  if (target == Target::STD)
    return std::move(res);

  // MLIR canonical dialects -> LLVM Dialect
  if (mlir::concretelang::pipeline::lowerStdToLLVMDialect(
          mlirContext, module, enablePass,
          this->loopParallelize || this->autoParallelize)
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

// Compile the source `s` to the target dialect `target`. If successful, the
// result can be retrieved using `getModule()` and `getLLVMModule()`,
// respectively depending on the target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(llvm::StringRef s, Target target, OptionalLib lib) {
  std::unique_ptr<llvm::MemoryBuffer> mb = llvm::MemoryBuffer::getMemBuffer(s);
  return this->compile(std::move(mb), target, lib);
}

// Compile the contained in `buffer` to the target dialect
// `target`. If successful, the result can be retrieved using
// `getModule()` and `getLLVMModule()`, respectively depending on the
// target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(std::unique_ptr<llvm::MemoryBuffer> buffer,
                        Target target, OptionalLib lib) {
  llvm::SourceMgr sm;

  sm.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  return this->compile(sm, target, lib);
}

template <class T>
llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(std::vector<T> inputs, std::string libraryPath) {
  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(libraryPath);
  auto target = CompilerEngine::Target::LIBRARY;

  for (auto input : inputs) {
    auto compilation = compile(input, target, outputLib);
    if (!compilation) {
      return StreamStringError("Can't compile: ")
             << llvm::toString(compilation.takeError());
    }
  }

  if (auto err = outputLib->emitArtifacts()) {
    return StreamStringError("Can't emit artifacts: ")
           << llvm::toString(std::move(err));
  }
  return *outputLib.get();
}

// explicit instantiation for a vector of string (for linking with lib/CAPI)
template llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(std::vector<std::string> inputs,
                        std::string libraryPath);

/** Returns the path of the shared library */
std::string CompilerEngine::Library::getSharedLibraryPath(std::string path) {
  return path + DOT_SHARED_LIB_EXT;
}

/** Returns the path of the static library */
std::string CompilerEngine::Library::getStaticLibraryPath(std::string path) {
  return path + DOT_STATIC_LIB_EXT;
}

/** Returns the path of the static library */
std::string CompilerEngine::Library::getClientParametersPath(std::string path) {
  return ClientParameters::getClientParametersPath(path);
}

const std::string CompilerEngine::Library::OBJECT_EXT = ".o";
const std::string CompilerEngine::Library::CLIENT_PARAMETERS_EXT =
    ".concrete.params.json";
const std::string CompilerEngine::Library::LINKER = "ld";
#ifdef __APPLE__
// ld in Mac can't find some symbols without specifying these libs
const std::string CompilerEngine::Library::LINKER_SHARED_OPT =
    " -dylib -lConcretelangRuntime -lc -o ";
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
  auto clientParamsPath = getClientParametersPath(libraryPath);
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
  auto libraryName = llvm::sys::path::filename(libraryPath).str();
  auto headerName = libraryName + "-client.h";
  auto headerPath = std::regex_replace(
      libraryPath, std::regex(libraryName + "$"), headerName);

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

  return headerPath;
}

llvm::Expected<std::string>
CompilerEngine::Library::addCompilation(CompilationResult &compilation) {
  llvm::Module *module = compilation.llvmModule.get();
  auto sourceName = module->getSourceFileName();
  if (sourceName == "" || sourceName == "LLVMDialectModule") {
    sourceName = this->libraryPath + ".module-" +
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

llvm::Expected<std::string> CompilerEngine::Library::emit(std::string dotExt,
                                                          std::string linker) {
  auto pathDotExt = ensureLibDotExt(libraryPath, dotExt);
  auto error = mlir::concretelang::emitLibrary(objectsPath, pathDotExt, linker);
  if (error) {
    return std::move(error);
  }
  return pathDotExt;
}

llvm::Expected<std::string> CompilerEngine::Library::emitShared() {
  auto path = emit(DOT_SHARED_LIB_EXT, LINKER + LINKER_SHARED_OPT);
  if (path) {
    sharedLibraryPath = path.get();
  }
  return path;
}

llvm::Expected<std::string> CompilerEngine::Library::emitStatic() {
  auto path = emit(DOT_STATIC_LIB_EXT, AR + AR_STATIC_OPT);
  if (path) {
    staticLibraryPath = path.get();
  }
  return path;
}

llvm::Error CompilerEngine::Library::emitArtifacts() {
  if (auto err = emitShared().takeError()) {
    return err;
  }
  if (auto err = emitStatic().takeError()) {
    return err;
  }
  if (auto err = emitClientParametersJSON().takeError()) {
    return err;
  }
  if (auto err = emitCppHeader().takeError()) {
    return err;
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
