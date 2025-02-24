// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_COMPILER_ENGINE_H
#define CONCRETELANG_SUPPORT_COMPILER_ENGINE_H

#include "concretelang/Common/Protocol.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/Support/ProgramInfoGeneration.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <optional>

using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {

/// Compilation context that acts as the root owner of LLVM and MLIR
/// data structures directly and indirectly referenced by artefacts
/// produced by the `CompilerEngine`.
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

enum Backend {
  CPU,
  GPU,
};

/// Compilation options allows to configure the compilation pipeline.
struct CompilationOptions {
  std::optional<mlir::concretelang::V0FHEConstraint> v0FHEConstraints;

  std::optional<mlir::concretelang::V0Parameter> v0Parameter;

  /// largeIntegerParameter force the compiler engine to lower FHE.eint using
  /// the large integers strategy with the given parameters.
  std::optional<mlir::concretelang::LargeIntegerParameter>
      largeIntegerParameter;

  bool verifyDiagnostics;

  /// Simulate options
  bool simulate;

  /// Enable overflow detection during simulation
  bool enableOverflowDetectionInSimulation;

  /// Parallelization options
  bool autoParallelize;
  bool loopParallelize;
  bool dataflowParallelize;

  /// Compression options
  bool compressEvaluationKeys;
  bool compressInputCiphertexts;

  /// Optimizer options
  optimizer::Config optimizerConfig;

  /// use GPU during execution by generating GPU operations if possible
  bool emitGPUOps;

  /// Other options
  bool batchTFHEOps;
  int64_t maxBatchSize;
  bool emitSDFGOps;
  bool unrollLoopsWithSDFGConvertibleOps;
  bool optimizeTFHE;

  std::optional<std::vector<int64_t>> fhelinalgTileSizes;

  /// When decomposing big integers into chunks, chunkSize is the total number
  /// of bits used for the message, including the carry, while chunkWidth is
  /// only the number of bits used during encoding and decoding of a big integer
  bool chunkIntegers;
  unsigned int chunkSize;
  unsigned int chunkWidth;

  /// When compiling from a dialect lower than FHE, one needs to provide
  /// encodings info manually to allow the client lib to be generated.
  std::optional<Message<concreteprotocol::ProgramEncodingInfo>> encodings;

  bool skipProgramInfo;

  bool enableTluFusing;
  bool printTluFusing;

  CompilationOptions()
      : v0FHEConstraints(std::nullopt), verifyDiagnostics(false),
        /// Simulate options
        simulate(false), enableOverflowDetectionInSimulation(false),
        // Parallelization options
        autoParallelize(false), loopParallelize(true),
        dataflowParallelize(false),
        /// Compression options
        compressEvaluationKeys(false), compressInputCiphertexts(false),
        /// Optimizer options
        optimizerConfig(optimizer::DEFAULT_CONFIG),
        /// GPU
        emitGPUOps(false),
        /// Other options
        batchTFHEOps(false), maxBatchSize(std::numeric_limits<int64_t>::max()),
        emitSDFGOps(false), unrollLoopsWithSDFGConvertibleOps(false),
        optimizeTFHE(true), chunkIntegers(false), chunkSize(4), chunkWidth(2),
        encodings(std::nullopt), enableTluFusing(true), printTluFusing(false){};

  /// @brief Constructor for CompilationOptions with default parameters for a
  /// specific backend.
  /// @param backend The backend to target.
  CompilationOptions(enum Backend backend) : CompilationOptions() {
    switch (backend) {
    case Backend::CPU:
      loopParallelize = true;
      break;
    case Backend::GPU:
      loopParallelize = true;
      batchTFHEOps = true;
      emitGPUOps = true;
      emitSDFGOps = true;
      char *env = getenv("SDFG_MAX_BATCH_SIZE");
      if (env != nullptr) {
        int64_t targetMax = strtoul(env, NULL, 10);
        if (targetMax > 0)
          maxBatchSize = targetMax;
      }
      break;
    }
  }
};

/// Set the global compilation options for the current compilation.
void setCurrentCompilationOptions(CompilationOptions options);

/// Get the global compilation options for the current compilation.
CompilationOptions getCurrentCompilationOptions();

/// Print table lookup fusing.
void printTluFusing(mlir::Value v1, mlir::Value v2, mlir::Value v1v2);

/// Result of an invocation of the `CompilerEngine` with optional
/// fields for the results produced by different stages.
class CompilationResult {
public:
  CompilationResult(std::shared_ptr<CompilationContext> compilationContext =
                        CompilationContext::createShared())
      : compilationContext(compilationContext) {}

  std::optional<mlir::OwningOpRef<mlir::ModuleOp>> mlirModuleRef;
  std::optional<Message<concreteprotocol::ProgramInfo>> programInfo;
  std::optional<ProgramCompilationFeedback> feedback;
  std::unique_ptr<llvm::Module> llvmModule;
  std::optional<mlir::concretelang::V0FHEContext> fheContext;

protected:
  std::shared_ptr<CompilationContext> compilationContext;
};

class Library {
  std::string outputDirPath;
  std::vector<std::string> objectsPath;
  /// Path to the runtime library. Will be linked to the output library if set
  std::string runtimeLibraryPath;
  bool cleanUp;
  mlir::concretelang::ProgramCompilationFeedback compilationFeedback;

public:
  std::optional<Message<concreteprotocol::ProgramInfo>> programInfo;

public:
  /// Create a library instance on which you can add compilation results.
  /// Then you can emit a library file with the given path.
  /// cleanUp at false keeps intermediate .obj files for later use.
  Library(std::string outputDirPath, std::string runtimeLibraryPath = "",
          bool cleanUp = true)
      : outputDirPath(outputDirPath), runtimeLibraryPath(runtimeLibraryPath),
        cleanUp(cleanUp), programInfo() {}
  /// Sets the compilation result used by the library
  llvm::Expected<std::string>
  setCompilationResult(CompilationResult &compilation);
  /// Emit the library artifacts with the previously added compilation result
  llvm::Error emitArtifacts(bool sharedLib, bool staticLib,
                            bool clientParameters, bool compilationFeedback);
  /// After a shared library has been emitted, its path is here
  std::string sharedLibraryPath;
  /// After a static library has been emitted, its path is here
  std::string staticLibraryPath;

  /// Returns the program info of the library.
  Result<Message<concreteprotocol::ProgramInfo>> getProgramInfo();

  /// Returns the path to the output dir.
  const std::string &getOutputDirPath() const;

  /// Returns the path of the shared library
  std::string getSharedLibraryPath() const;

  /// Returns the path of the static library
  std::string getStaticLibraryPath() const;

  /// Returns the path of the program info
  std::string getProgramInfoPath() const;

  /// Returns the path of the compilation feedback
  std::string getCompilationFeedbackPath() const;

  // For advanced use
  const static std::string OBJECT_EXT, LINKER, LINKER_SHARED_OPT, AR,
      AR_STATIC_OPT, DOT_STATIC_LIB_EXT, DOT_SHARED_LIB_EXT;
  void addExtraObjectFilePath(std::string objectFilePath);
  llvm::Expected<std::string>
  emit(std::string path, std::string dotExt, std::string linker,
       std::optional<std::vector<std::string>> extraArgs = {});
  ~Library();

private:
  /// Emit a shared library with the previously added compilation result
  llvm::Expected<std::string> emitStatic();
  /// Emit a shared library with the previously added compilation result
  llvm::Expected<std::string> emitShared();
  /// Emit a json ProgramInfo corresponding to library content
  llvm::Expected<std::string> emitProgramInfoJSON();
  /// Emit a json CompilationFeedback corresponding to library content
  llvm::Expected<std::string> emitCompilationFeedbackJSON();
};

/// Specification of the exit stage of the compilation pipeline
enum class Target {
  /// Only read sources and produce corresponding MLIR module
  ROUND_TRIP,

  /// Read sources and exit before any lowering
  FHE,

  /// Read sources, lower all FHELinalg operations to operations
  /// from the Linalg dialect
  FHE_LINALG_GENERIC,

  /// Read sources and lower all the FHELinalg operations to FHE
  /// operations, dump after data-flow parallelization
  FHE_DF_PARALLELIZED,

  /// Read sources and lower all the FHELinalg operations to FHE operations
  /// and scf loops
  FHE_NO_LINALG,

  /// Read sources and lower all FHE operations to unparameterized TFHE
  /// operations
  TFHE,

  /// Read sources and lower all FHE operations to TFHE
  /// operations, then parametrize the TFHE operations
  PARAMETRIZED_TFHE,

  /// Batch TFHE operations
  BATCHED_TFHE,

  /// Read sources and lower all FHE operations to normalized TFHE
  /// operations
  NORMALIZED_TFHE,

  /// Read sources and lower all FHE operations to simulated TFHE
  SIMULATED_TFHE,

  /// Read sources and lower all FHE and TFHE operations to Concrete
  /// operations
  CONCRETE,

  /// Read sources and lower all FHE and TFHE operations to Concrete
  /// then extract SDFG operations
  SDFG,

  /// Read sources and lower all FHE, TFHE and Concrete
  /// operations to canonical MLIR dialects. Cryptographic operations
  /// are lowered to invocations of the concrete library.
  STD,

  /// Read sources and lower all FHE, TFHE and Concrete
  /// operations to operations from the LLVM dialect. Cryptographic
  /// operations are lowered to invocations of the concrete library.
  LLVM,

  /// Same as `LLVM`, but lowers to actual LLVM IR instead of the
  /// LLVM dialect
  LLVM_IR,

  /// Same as `LLVM_IR`, but invokes the LLVM optimization pipeline
  /// to produce optimized LLVM IR
  OPTIMIZED_LLVM_IR,

  /// Same as `OPTIMIZED_LLVM_IR`, but compiles and add an object file to a
  /// futur library
  LIBRARY
};

class CompilerEngine {
public:
  CompilerEngine(std::shared_ptr<CompilationContext> compilationContext)
      : overrideMaxEintPrecision(), overrideMaxMANP(), compilerOptions(),
        generateProgramInfo(true),
        enablePass([](mlir::Pass *pass) { return true; }),
        compilationContext(compilationContext) {}

  llvm::Expected<CompilationResult>
  compile(llvm::StringRef s, Target target,
          std::optional<std::shared_ptr<Library>> lib = {});

  llvm::Expected<CompilationResult>
  compile(std::unique_ptr<llvm::MemoryBuffer> buffer, Target target,
          std::optional<std::shared_ptr<Library>> lib = {});

  llvm::Expected<CompilationResult>
  compile(llvm::SourceMgr &sm, Target target,
          std::optional<std::shared_ptr<Library>> lib = {});

  llvm::Expected<CompilationResult>
  compile(mlir::ModuleOp module, Target target,
          std::optional<std::shared_ptr<Library>> lib = {});

  llvm::Expected<Library>
  compile(std::vector<std::string> inputs, std::string outputDirPath,
          std::string runtimeLibraryPath = "", bool generateSharedLib = true,
          bool generateStaticLib = true, bool generateClientParameters = true,
          bool generateCompilationFeedback = true);

  /// Compile and emit artifact to the given outputDirPath from an LLVM source
  /// manager.
  llvm::Expected<Library>
  compile(llvm::SourceMgr &sm, std::string outputDirPath,
          std::string runtimeLibraryPath = "", bool generateSharedLib = true,
          bool generateStaticLib = true, bool generateClientParameters = true,
          bool generateCompilationFeedback = true);

  llvm::Expected<Library>
  compile(mlir::ModuleOp module, std::string outputDirPath,
          std::string runtimeLibraryPath = "", bool generateSharedLib = true,
          bool generateStaticLib = true, bool generateClientParameters = true,
          bool generateCompilationFeedback = true);

  void setCompilationOptions(CompilationOptions options) {
    setCurrentCompilationOptions(options);
    compilerOptions = std::move(options);
    if (compilerOptions.v0FHEConstraints.has_value()) {
      setFHEConstraints(*compilerOptions.v0FHEConstraints);
    }
  }

  CompilationOptions &getCompilationOptions() { return compilerOptions; }

  void setFHEConstraints(const mlir::concretelang::V0FHEConstraint &c);
  void setMaxEintPrecision(size_t v);
  void setMaxMANP(size_t v);
  void setGenerateProgramInfo(bool v);
  void setEnablePass(std::function<bool(mlir::Pass *)> enablePass);

protected:
  std::optional<size_t> overrideMaxEintPrecision;
  std::optional<size_t> overrideMaxMANP;
  CompilationOptions compilerOptions;
  bool generateProgramInfo;
  std::function<bool(mlir::Pass *)> enablePass;

  std::shared_ptr<CompilationContext> compilationContext;

private:
  llvm::Expected<std::optional<optimizer::Description>>
  getConcreteOptimizerDescription(CompilationResult &res);
  llvm::Error determineFHEParameters(CompilationResult &res);
  mlir::LogicalResult
  materializeOptimizerPartitionFrontiers(CompilationResult &res);
};

} // namespace concretelang
} // namespace mlir

#endif
