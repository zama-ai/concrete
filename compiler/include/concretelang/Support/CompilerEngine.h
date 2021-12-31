// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_SUPPORT_COMPILER_ENGINE_H
#define CONCRETELANG_SUPPORT_COMPILER_ENGINE_H

#include <concretelang/Conversion/Utils/GlobalFHEContext.h>
#include <concretelang/Support/ClientParameters.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace concretelang {

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
    llvm::Optional<mlir::concretelang::ClientParameters> clientParameters;
    std::unique_ptr<llvm::Module> llvmModule;
    llvm::Optional<mlir::concretelang::V0FHEContext> fheContext;

  protected:
    std::shared_ptr<CompilationContext> compilationContext;
  };

  class Library {
    std::string libraryPath;
    std::vector<std::string> objectsPath;
    bool cleanUp;

  public:
    /** Create a library instance on which you can add compilation results.
     * Then you can emit a library file with the given path.
     * cleanUp at false keeps intermediate .obj files for later use. */
    Library(std::string libraryPath, bool cleanUp = true)
        : libraryPath(libraryPath), cleanUp(cleanUp) {}
    /** Add a compilation result to the library */
    llvm::Expected<std::string> addCompilation(CompilationResult &compilation);
    /** Emit a shared library with the previously added compilation result */
    llvm::Expected<std::string> emitShared();
    /** Emit a shared library with the previously added compilation result */
    llvm::Expected<std::string> emitStatic();
    /** After a shared library has been emitted, its path is here */
    std::string sharedLibraryPath;
    /** After a static library has been emitted, its path is here */
    std::string staticLibraryPath;

    // For advanced use
    const static std::string OBJECT_EXT, LINKER, LINKER_SHARED_OPT, AR,
        AR_STATIC_OPT, DOT_STATIC_LIB_EXT, DOT_SHARED_LIB_EXT;
    void addExtraObjectFilePath(std::string objectFilePath);
    llvm::Expected<std::string> emit(std::string dotExt, std::string linker);
    ~Library();
  };

  // Specification of the exit stage of the compilation pipeline
  enum class Target {
    // Only read sources and produce corresponding MLIR module
    ROUND_TRIP,

    // Read sources and exit before any lowering
    FHE,

    // Read sources and lower all FHE operations to TFHE
    // operations
    TFHE,

    // Read sources and lower all FHE and TFHE operations to Concrete
    // operations
    CONCRETE,

    // Read sources and lower all FHE, TFHE and Concrete
    // operations to canonical MLIR dialects. Cryptographic operations
    // are lowered to invocations of the concrete library.
    STD,

    // Read sources and lower all FHE, TFHE and Concrete
    // operations to operations from the LLVM dialect. Cryptographic
    // operations are lowered to invocations of the concrete library.
    LLVM,

    // Same as `LLVM`, but lowers to actual LLVM IR instead of the
    // LLVM dialect
    LLVM_IR,

    // Same as `LLVM_IR`, but invokes the LLVM optimization pipeline
    // to produce optimized LLVM IR
    OPTIMIZED_LLVM_IR,

    // Same as `OPTIMIZED_LLVM_IR`, but compiles and add an object file to a
    // futur library
    LIBRARY
  };

  CompilerEngine(std::shared_ptr<CompilationContext> compilationContext)
      : overrideMaxEintPrecision(), overrideMaxMANP(),
        clientParametersFuncName(), verifyDiagnostics(false),
        autoParallelize(false), generateClientParameters(false),
        enablePass([](mlir::Pass *pass) { return true; }),
        compilationContext(compilationContext) {}

  llvm::Expected<CompilationResult>
  compile(llvm::StringRef s, Target target,
          llvm::Optional<std::shared_ptr<Library>> lib = {});

  llvm::Expected<CompilationResult>
  compile(std::unique_ptr<llvm::MemoryBuffer> buffer, Target target,
          llvm::Optional<std::shared_ptr<Library>> lib = {});

  llvm::Expected<CompilationResult>
  compile(llvm::SourceMgr &sm, Target target,
          llvm::Optional<std::shared_ptr<Library>> lib = {});

  template <class T>
  llvm::Expected<CompilerEngine::Library> compile(std::vector<T> inputs,
                                                  std::string libraryPath);

  void setFHEConstraints(const mlir::concretelang::V0FHEConstraint &c);
  void setMaxEintPrecision(size_t v);
  void setMaxMANP(size_t v);
  void setVerifyDiagnostics(bool v);
  void setAutoParallelize(bool v);
  void setGenerateClientParameters(bool v);
  void setClientParametersFuncName(const llvm::StringRef &name);
  void setFHELinalgTileSizes(llvm::ArrayRef<int64_t> sizes);
  void setEnablePass(std::function<bool(mlir::Pass *)> enablePass);

protected:
  llvm::Optional<size_t> overrideMaxEintPrecision;
  llvm::Optional<size_t> overrideMaxMANP;
  llvm::Optional<std::string> clientParametersFuncName;
  llvm::Optional<std::vector<int64_t>> fhelinalgTileSizes;

  bool verifyDiagnostics;
  bool autoParallelize;
  bool generateClientParameters;
  std::function<bool(mlir::Pass *)> enablePass;

  std::shared_ptr<CompilationContext> compilationContext;

private:
  llvm::Expected<llvm::Optional<mlir::concretelang::V0FHEConstraint>>
  getV0FHEConstraint(CompilationResult &res);
  llvm::Error determineFHEParameters(CompilationResult &res);
};

} // namespace concretelang
} // namespace mlir

#endif
