#include <llvm/Support/Error.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <zamalang/Dialect/HLFHE/IR/HLFHEDialect.h>
#include <zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h>
#include <zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h>
#include <zamalang/Support/CompilerEngine.h>
#include <zamalang/Support/Error.h>
#include <zamalang/Support/Jit.h>
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

void CompilerEngine::setGenerateKeySet(bool v) { this->generateKeySet = v; }

void CompilerEngine::setGenerateClientParameters(bool v) {
  this->generateClientParameters = v;
}

void CompilerEngine::setMaxEintPrecision(size_t v) {
  this->overrideMaxEintPrecision = v;
}

void CompilerEngine::setParametrizeMidLFHE(bool v) {
  this->parametrizeMidLFHE = v;
}

void CompilerEngine::setMaxMANP(size_t v) { this->overrideMaxMANP = v; }

void CompilerEngine::setClientParametersFuncName(const llvm::StringRef &name) {
  this->clientParametersFuncName = name.str();
}

// Helper function detecting the FHE dialect with the highest level of
// abstraction used in `module`. If no FHE dialect is used, the
// function returns `CompilerEngine::FHEDialect::NONE`.
CompilerEngine::FHEDialect
CompilerEngine::detectHighestFHEDialect(mlir::ModuleOp module) {
  CompilerEngine::FHEDialect highestDialect = CompilerEngine::FHEDialect::NONE;

  mlir::TypeID hlfheID =
      mlir::TypeID::get<mlir::zamalang::HLFHE::HLFHEDialect>();
  mlir::TypeID midlfheID =
      mlir::TypeID::get<mlir::zamalang::MidLFHE::MidLFHEDialect>();
  mlir::TypeID lowlfheID =
      mlir::TypeID::get<mlir::zamalang::LowLFHE::LowLFHEDialect>();

  // Helper lambda updating the currently highest dialect if necessary
  // by dialect type ID
  auto updateDialectFromDialectID = [&](mlir::TypeID dialectID) {
    if (dialectID == hlfheID) {
      highestDialect = CompilerEngine::FHEDialect::HLFHE;
      return true;
    } else if (dialectID == lowlfheID &&
               highestDialect == CompilerEngine::FHEDialect::NONE) {
      highestDialect = CompilerEngine::FHEDialect::LOWLFHE;
    } else if (dialectID == midlfheID &&
               (highestDialect == CompilerEngine::FHEDialect::NONE ||
                highestDialect == CompilerEngine::FHEDialect::LOWLFHE)) {
      highestDialect = CompilerEngine::FHEDialect::MIDLFHE;
    }

    return false;
  };

  // Helper lambda updating the currently highest dialect if necessary
  // by value type
  std::function<bool(mlir::Type)> updateDialectFromType =
      [&](mlir::Type ty) -> bool {
    if (updateDialectFromDialectID(ty.getDialect().getTypeID()))
      return true;

    if (mlir::TensorType tensorTy = ty.dyn_cast_or_null<mlir::TensorType>())
      return updateDialectFromType(tensorTy.getElementType());

    return false;
  };

  module.walk([&](mlir::Operation *op) {
    // Check operation itself
    if (updateDialectFromDialectID(op->getDialect()->getTypeID()))
      return mlir::WalkResult::interrupt();

    // Check types of operands
    for (mlir::Value operand : op->getOperands()) {
      if (updateDialectFromType(operand.getType()))
        return mlir::WalkResult::interrupt();
    }

    // Check types of results
    for (mlir::Value res : op->getResults()) {
      if (updateDialectFromType(res.getType())) {
        return mlir::WalkResult::interrupt();
      }
    }

    return mlir::WalkResult::advance();
  });

  return highestDialect;
}

// Sets the FHE parameters of `res` either through autodetection or
// fixed constraints provided in
// `CompilerEngine::overrideMaxEintPrecision` and
// `CompilerEngine::overrideMaxMANP`.
//
// Autodetected values can be partially or fully overridden through
// `CompilerEngine::overrideMaxEintPrecision` and
// `CompilerEngine::overrideMaxMANP`.
//
// If `noOverrideAutodetected` is true, autodetected values are not
// overriden and used directly for `res`.
//
// Return an error if autodetection fails.
llvm::Error
CompilerEngine::determineFHEParameters(CompilationResult &res,
                                       bool noOverrideAutodetected) {
  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();
  mlir::ModuleOp module = res.mlirModuleRef->get();
  llvm::Optional<mlir::zamalang::V0FHEConstraint> fheConstraints;

  // Determine FHE constraints either through autodetection or through
  // overridden values
  if (this->overrideMaxEintPrecision.hasValue() &&
      this->overrideMaxMANP.hasValue() && !noOverrideAutodetected) {
    fheConstraints.emplace(mlir::zamalang::V0FHEConstraint{
        this->overrideMaxMANP.getValue(),
        this->overrideMaxEintPrecision.getValue()});

  } else {
    llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
        fheConstraintsOrErr =
            mlir::zamalang::pipeline::getFHEConstraintsFromHLFHE(mlirContext,
                                                                 module);

    if (auto err = fheConstraintsOrErr.takeError())
      return std::move(err);

    if (!fheConstraintsOrErr.get().hasValue()) {
      return StreamStringError("Could not determine maximum required precision "
                               "for encrypted integers and maximum value for "
                               "the Minimal Arithmetic Noise Padding");
    }

    if (noOverrideAutodetected)
      return llvm::Error::success();

    fheConstraints = fheConstraintsOrErr.get();

    // Override individual values if requested
    if (this->overrideMaxEintPrecision.hasValue())
      fheConstraints->p = this->overrideMaxEintPrecision.getValue();

    if (this->overrideMaxMANP.hasValue())
      fheConstraints->norm2 = this->overrideMaxMANP.getValue();
  }

  const mlir::zamalang::V0Parameter *fheParams =
      getV0Parameter(fheConstraints.getValue());

  if (!fheParams) {
    return StreamStringError()
           << "Could not determine V0 parameters for 2-norm of "
           << fheConstraints->norm2 << " and p of " << fheConstraints->p;
  }

  res.fheContext.emplace(
      mlir::zamalang::V0FHEContext{*fheConstraints, *fheParams});

  return llvm::Error::success();
}

// Performs all lowering from HLFHE to the FHE dialect with the lwoest
// level of abstraction that requires FHE parameters.
//
// Returns an error if any of the lowerings fails.
llvm::Error CompilerEngine::lowerParamDependentHalf(Target target,
                                                    CompilationResult &res) {
  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();
  mlir::ModuleOp module = res.mlirModuleRef->get();

  // HLFHE -> MidLFHE
  if (mlir::zamalang::pipeline::lowerHLFHEToMidLFHE(mlirContext, module, false)
          .failed()) {
    return StreamStringError("Lowering from HLFHE to MidLFHE failed");
  }

  if (target == Target::MIDLFHE)
    return llvm::Error::success();

  // MidLFHE -> LowLFHE
  if (mlir::zamalang::pipeline::lowerMidLFHEToLowLFHE(
          mlirContext, module, *res.fheContext, this->parametrizeMidLFHE)
          .failed()) {
    return StreamStringError("Lowering from MidLFHE to LowLFHE failed");
  }

  return llvm::Error::success();
}

// Compile the sources managed by the source manager `sm` to the
// target dialect `target`. If successful, the result can be retrieved
// using `getModule()` and `getLLVMModule()`, respectively depending
// on the target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(llvm::SourceMgr &sm, Target target) {
  CompilationResult res(this->compilationContext);

  mlir::MLIRContext &mlirContext = *this->compilationContext->getMLIRContext();

  mlir::SourceMgrDiagnosticVerifierHandler smHandler(sm, &mlirContext);
  mlirContext.printOpOnDiagnostic(false);

  mlir::OwningModuleRef mlirModuleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sm, &mlirContext);

  if (this->verifyDiagnostics) {
    if (smHandler.verify().failed())
      return StreamStringError("Verification of diagnostics failed");
    else
      return res;
  }

  if (!mlirModuleRef)
    return StreamStringError("Could not parse source");

  res.mlirModuleRef = std::move(mlirModuleRef);
  mlir::ModuleOp module = res.mlirModuleRef->get();

  if (target == Target::HLFHE || target == Target::ROUND_TRIP)
    return res;

  // Detect highest FHE dialect and check if FHE parameter
  // autodetection / lowering of parameter-dependent dialects can be
  // skipped
  FHEDialect highestFHEDialect = this->detectHighestFHEDialect(module);

  if (highestFHEDialect == FHEDialect::HLFHE ||
      highestFHEDialect == FHEDialect::MIDLFHE ||
      this->generateClientParameters) {
    bool noOverrideAutoDetected = (target == Target::HLFHE_MANP);
    if (auto err = this->determineFHEParameters(res, noOverrideAutoDetected))
      return std::move(err);
  }

  // return early if only the MANP pass was requested
  if (target == Target::HLFHE_MANP)
    return res;

  if (highestFHEDialect == FHEDialect::HLFHE ||
      highestFHEDialect == FHEDialect::MIDLFHE) {
    if (llvm::Error err = this->lowerParamDependentHalf(target, res))
      return std::move(err);
  }

  if (target == Target::HLFHE_MANP || target == Target::MIDLFHE ||
      target == Target::LOWLFHE)
    return res;

  // LowLFHE -> Canonical dialects
  if (mlir::zamalang::pipeline::lowerLowLFHEToStd(mlirContext, module)
          .failed()) {
    return StreamStringError(
        "Lowering from LowLFHE to canonical MLIR dialects failed");
  }

  if (target == Target::STD)
    return res;

  // Generate client parameters if requested
  if (this->generateClientParameters) {
    if (!this->clientParametersFuncName.hasValue()) {
      return StreamStringError(
          "Generation of client parameters requested, but no function name "
          "specified");
    }

    llvm::Expected<mlir::zamalang::ClientParameters> clientParametersOrErr =
        mlir::zamalang::createClientParametersForV0(
            *res.fheContext, *this->clientParametersFuncName, module);

    if (llvm::Error err = clientParametersOrErr.takeError())
      return std::move(err);

    res.clientParameters = clientParametersOrErr.get();
  }

  // Generate Key set if requested
  if (this->generateKeySet) {
    if (!res.clientParameters.hasValue()) {
      return StreamStringError("Generation of keyset requested without request "
                               "for generation of client parameters");
    }

    llvm::Expected<std::unique_ptr<mlir::zamalang::KeySet>> keySetOrErr =
        mlir::zamalang::KeySet::generate(*res.clientParameters, 0, 0);

    if (auto err = keySetOrErr.takeError())
      return std::move(err);

    res.keySet = std::move(*keySetOrErr);
  }

  // MLIR canonical dialects -> LLVM Dialect
  if (mlir::zamalang::pipeline::lowerStdToLLVMDialect(mlirContext, module,
                                                      false)
          .failed()) {
    return StreamStringError("Failed to lower to LLVM dialect");
  }

  if (target == Target::LLVM)
    return res;

  // Lowering to actual LLVM IR (i.e., not the LLVM dialect)
  llvm::LLVMContext &llvmContext = *this->compilationContext->getLLVMContext();

  res.llvmModule = mlir::zamalang::pipeline::lowerLLVMDialectToLLVMIR(
      mlirContext, llvmContext, module);

  if (!res.llvmModule)
    return StreamStringError("Failed to convert from LLVM dialect to LLVM IR");

  if (target == Target::LLVM_IR)
    return res;

  if (mlir::zamalang::pipeline::optimizeLLVMModule(llvmContext, *res.llvmModule)
          .failed()) {
    return StreamStringError("Failed to optimize LLVM IR");
  }

  if (target == Target::OPTIMIZED_LLVM_IR)
    return res;

  return res;
} // namespace zamalang

// Compile the source `s` to the target dialect `target`. If successful, the
// result can be retrieved using `getModule()` and `getLLVMModule()`,
// respectively depending on the target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(llvm::StringRef s, Target target) {
  std::unique_ptr<llvm::MemoryBuffer> mb = llvm::MemoryBuffer::getMemBuffer(s);
  llvm::Expected<CompilationResult> res = this->compile(std::move(mb), target);

  return std::move(res);
}

// Compile the contained in `buffer` to the target dialect
// `target`. If successful, the result can be retrieved using
// `getModule()` and `getLLVMModule()`, respectively depending on the
// target dialect.
llvm::Expected<CompilerEngine::CompilationResult>
CompilerEngine::compile(std::unique_ptr<llvm::MemoryBuffer> buffer,
                        Target target) {
  llvm::SourceMgr sm;

  sm.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  llvm::Expected<CompilationResult> res = this->compile(sm, target);

  return std::move(res);
}

} // namespace zamalang
} // namespace mlir
