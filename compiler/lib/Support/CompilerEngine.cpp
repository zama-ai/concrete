#include <stdio.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <zamalang/Dialect/HLFHE/IR/HLFHEDialect.h>
#include <zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h>
#include <zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h>
#include <zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h>
#include <zamalang/Support/CompilerEngine.h>
#include <zamalang/Support/Error.h>
#include <zamalang/Support/Jit.h>
#include <zamalang/Support/LLVMEmitFile.h>
#include <zamalang/Support/Pipeline.h>

namespace mlir {
namespace zamalang {

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

    this->mlirContext->getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::zamalang::HLFHELinalg::HLFHELinalgDialect>();
    this->mlirContext
        ->getOrLoadDialect<mlir::zamalang::LowLFHE::LowLFHEDialect>();
    this->mlirContext->getOrLoadDialect<mlir::StandardOpsDialect>();
    this->mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();
    this->mlirContext->getOrLoadDialect<mlir::linalg::LinalgDialect>();
    this->mlirContext->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
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
    const mlir::zamalang::V0FHEConstraint &c) {
  this->overrideMaxEintPrecision = c.p;
  this->overrideMaxMANP = c.norm2;
}

void CompilerEngine::setVerifyDiagnostics(bool v) {
  this->verifyDiagnostics = v;
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

void CompilerEngine::setEnablePass(
    std::function<bool(mlir::Pass *)> enablePass) {
  this->enablePass = enablePass;
}

// Returns the overwritten V0FHEConstraint or try to compute them from HLFHE
llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
CompilerEngine::getV0FHEConstraint(CompilationResult &res) {
  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();
  mlir::ModuleOp module = res.mlirModuleRef->get();
  llvm::Optional<mlir::zamalang::V0FHEConstraint> fheConstraints;
  // If the values has been overwritten returns
  if (this->overrideMaxEintPrecision.hasValue() &&
      this->overrideMaxMANP.hasValue()) {
    return mlir::zamalang::V0FHEConstraint{
        this->overrideMaxMANP.getValue(),
        this->overrideMaxEintPrecision.getValue()};
  }
  // Else compute constraint from HLFHE
  llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
      fheConstraintsOrErr =
          mlir::zamalang::pipeline::getFHEConstraintsFromHLFHE(
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
  const mlir::zamalang::V0Parameter *fheParams =
      getV0Parameter(fheConstraintOrErr.get().getValue());

  if (!fheParams) {
    return StreamStringError()
           << "Could not determine V0 parameters for 2-norm of "
           << (*fheConstraintOrErr)->norm2 << " and p of "
           << (*fheConstraintOrErr)->p;
  }
  res.fheContext.emplace(mlir::zamalang::V0FHEContext{
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

  // HLFHE High level pass to determine FHE parameters
  if (auto err = this->determineFHEParameters(res))
    return std::move(err);
  if (target == Target::HLFHE)
    return std::move(res);

  // HLFHE -> MidLFHE
  if (mlir::zamalang::pipeline::lowerHLFHEToMidLFHE(mlirContext, module,
                                                    enablePass)
          .failed()) {
    return errorDiag("Lowering from HLFHE to MidLFHE failed");
  }
  if (target == Target::MIDLFHE)
    return std::move(res);

  // MidLFHE -> LowLFHE
  if (mlir::zamalang::pipeline::lowerMidLFHEToLowLFHE(
          mlirContext, module, res.fheContext, this->enablePass)
          .failed()) {
    return errorDiag("Lowering from MidLFHE to LowLFHE failed");
  }
  if (target == Target::LOWLFHE)
    return std::move(res);

  // LowLFHE -> Canonical dialects
  if (mlir::zamalang::pipeline::lowerLowLFHEToStd(mlirContext, module,
                                                  enablePass)
          .failed()) {
    return errorDiag("Lowering from LowLFHE to canonical MLIR dialects failed");
  }
  if (target == Target::STD)
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

    llvm::Expected<mlir::zamalang::ClientParameters> clientParametersOrErr =
        mlir::zamalang::createClientParametersForV0(
            *res.fheContext, *this->clientParametersFuncName, module);

    if (llvm::Error err = clientParametersOrErr.takeError())
      return std::move(err);

    res.clientParameters = clientParametersOrErr.get();
  }

  // MLIR canonical dialects -> LLVM Dialect
  if (mlir::zamalang::pipeline::lowerStdToLLVMDialect(mlirContext, module,
                                                      enablePass)
          .failed()) {
    return errorDiag("Failed to lower to LLVM dialect");
  }

  if (target == Target::LLVM)
    return std::move(res);

  // Lowering to actual LLVM IR (i.e., not the LLVM dialect)
  llvm::LLVMContext &llvmContext = *this->compilationContext->getLLVMContext();

  res.llvmModule = mlir::zamalang::pipeline::lowerLLVMDialectToLLVMIR(
      mlirContext, llvmContext, module);

  if (!res.llvmModule)
    return StreamStringError("Failed to convert from LLVM dialect to LLVM IR");

  if (target == Target::LLVM_IR)
    return std::move(res);

  if (mlir::zamalang::pipeline::optimizeLLVMModule(llvmContext, *res.llvmModule)
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
  llvm::Expected<CompilationResult> res =
      this->compile(std::move(mb), target, lib);

  return std::move(res);
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

  llvm::Expected<CompilationResult> res = this->compile(sm, target, lib);

  return std::move(res);
}

template <class T>
llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(std::vector<T> inputs, std::string libraryPath) {
  using Library = mlir::zamalang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(libraryPath);
  auto target = CompilerEngine::Target::LIBRARY;

  for (auto input : inputs) {
    auto compilation = compile(input, target, outputLib);
    if (!compilation) {
      return StreamStringError("Can't compile: ")
             << llvm::toString(compilation.takeError());
    }
  }

  auto libPath = outputLib->emitShared();
  if (!libPath) {
    return StreamStringError("Can't link: ")
           << llvm::toString(libPath.takeError());
  }
  return *outputLib.get();
}

// explicit instantiation for a vector of string (for linking with lib/CAPI)
template llvm::Expected<CompilerEngine::Library>
CompilerEngine::compile(std::vector<std::string> inputs,
                        std::string libraryPath);

const std::string CompilerEngine::Library::OBJECT_EXT = ".o";
const std::string CompilerEngine::Library::LINKER = "ld";
const std::string CompilerEngine::Library::LINKER_SHARED_OPT = " --shared -o ";
const std::string CompilerEngine::Library::AR = "ar";
const std::string CompilerEngine::Library::AR_STATIC_OPT = " rcs ";
const std::string CompilerEngine::Library::DOT_STATIC_LIB_EXT = ".a";
const std::string CompilerEngine::Library::DOT_SHARED_LIB_EXT = ".so";

void CompilerEngine::Library::addExtraObjectFilePath(std::string path) {
  objectsPath.push_back(path);
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
  auto error = mlir::zamalang::emitObject(*module, objectPath);

  if (!error) {
    addExtraObjectFilePath(objectPath);
    return objectPath;
  }

  return error;
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
  auto error = mlir::zamalang::emitLibrary(objectsPath, pathDotExt, linker);
  if (error) {
    return std::move(error);
  } else {
    return pathDotExt;
  }
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

CompilerEngine::Library::~Library() {
  if (cleanUp) {
    for (auto path : objectsPath) {
      remove(path.c_str());
    }
  }
}

} // namespace zamalang
} // namespace mlir
