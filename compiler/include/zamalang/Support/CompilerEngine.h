#ifndef ZAMALANG_SUPPORT_COMPILER_ENGINE_H
#define ZAMALANG_SUPPORT_COMPILER_ENGINE_H

#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <zamalang/Conversion/Utils/GlobalFHEContext.h>
#include <zamalang/Support/ClientParameters.h>

namespace mlir {
namespace zamalang {

// Compilation context that acts as the root owner of LLVM and MLIR
// data structures directly and indirectly referenced by artefacts
// produced by the `CompilerEngine`.
class CompilationContext {
public:
  CompilationContext();
  ~CompilationContext();

  mlir::MLIRContext *getMLIRContext();
  llvm::LLVMContext *getLLVMContext();

  static std::shared_ptr<CompilationContext> createShared();

protected:
  mlir::MLIRContext *mlirContext;
  llvm::LLVMContext *llvmContext;
};

class CompilerEngine {
public:
  // Result of an invocation of the `CompilerEngine` with optional
  // fields for the results produced by different stages.
  class CompilationResult {
  public:
    CompilationResult(std::shared_ptr<CompilationContext> compilationContext =
                          CompilationContext::createShared())
        : compilationContext(compilationContext) {}

    llvm::Optional<mlir::OwningModuleRef> mlirModuleRef;
    llvm::Optional<mlir::zamalang::ClientParameters> clientParameters;
    std::unique_ptr<llvm::Module> llvmModule;
    llvm::Optional<mlir::zamalang::V0FHEContext> fheContext;

  protected:
    std::shared_ptr<CompilationContext> compilationContext;
  };

  // Specification of the exit stage of the compilation pipeline
  enum class Target {
    // Only read sources and produce corresponding MLIR module
    ROUND_TRIP,

    // Read sources and exit before any lowering
    HLFHE,

    // Read sources and attempt to run the Minimal Arithmetic Noise
    // Padding pass
    HLFHE_MANP,

    // Read sources and lower all HLFHE operations to MidLFHE
    // operations
    MIDLFHE,

    // Read sources and lower all HLFHE and MidLFHE operations to LowLFHE
    // operations
    LOWLFHE,

    // Read sources and lower all HLFHE, MidLFHE and LowLFHE
    // operations to canonical MLIR dialects. Cryptographic operations
    // are lowered to invocations of the concrete library.
    STD,

    // Read sources and lower all HLFHE, MidLFHE and LowLFHE
    // operations to operations from the LLVM dialect. Cryptographic
    // operations are lowered to invocations of the concrete library.
    LLVM,

    // Same as `LLVM`, but lowers to actual LLVM IR instead of the
    // LLVM dialect
    LLVM_IR,

    // Same as `LLVM_IR`, but invokes the LLVM optimization pipeline
    // to produce optimized LLVM IR
    OPTIMIZED_LLVM_IR
  };

  CompilerEngine(std::shared_ptr<CompilationContext> compilationContext)
      : overrideMaxEintPrecision(), overrideMaxMANP(),
        clientParametersFuncName(), verifyDiagnostics(false),
        generateClientParameters(false), parametrizeMidLFHE(true),
        compilationContext(compilationContext) {}

  llvm::Expected<CompilationResult> compile(llvm::StringRef s, Target target);

  llvm::Expected<CompilationResult>
  compile(std::unique_ptr<llvm::MemoryBuffer> buffer, Target target);

  llvm::Expected<CompilationResult> compile(llvm::SourceMgr &sm, Target target);

  void setFHEConstraints(const mlir::zamalang::V0FHEConstraint &c);
  void setMaxEintPrecision(size_t v);
  void setMaxMANP(size_t v);
  void setVerifyDiagnostics(bool v);
  void setGenerateClientParameters(bool v);
  void setParametrizeMidLFHE(bool v);
  void setClientParametersFuncName(const llvm::StringRef &name);

protected:
  llvm::Optional<size_t> overrideMaxEintPrecision;
  llvm::Optional<size_t> overrideMaxMANP;
  llvm::Optional<std::string> clientParametersFuncName;
  bool verifyDiagnostics;
  bool generateClientParameters;
  bool parametrizeMidLFHE;

  std::shared_ptr<CompilationContext> compilationContext;

  // Helper enum identifying an FHE dialect (`HLFHE`, `MIDLFHE`, `LOWLFHE`)
  // or indicating that no FHE dialect is used (`NONE`).
  enum class FHEDialect { HLFHE, MIDLFHE, LOWLFHE, NONE };
  static FHEDialect detectHighestFHEDialect(mlir::ModuleOp module);

private:
  llvm::Error lowerParamDependentHalf(Target target, CompilationResult &res);
  llvm::Error determineFHEParameters(CompilationResult &res, bool noOverride);
};

} // namespace zamalang
} // namespace mlir

#endif
