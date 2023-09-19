// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>

#include "capnp/compat/json.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/LLVMEmitFile.h"
#include "concretelang/Support/Pipeline.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/Support/logging.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using concretelang::keysets::Keyset;
namespace encodings = mlir::concretelang::encodings;
namespace optimizer = mlir::concretelang::optimizer;

enum Action {
  ROUND_TRIP,
  DUMP_FHE,
  DUMP_FHE_LINALG_GENERIC,
  DUMP_FHE_NO_LINALG,
  DUMP_TFHE,
  DUMP_NORMALIZED_TFHE,
  DUMP_PARAMETRIZED_TFHE,
  DUMP_BATCHED_TFHE,
  DUMP_SIMULATED_TFHE,
  DUMP_CONCRETE,
  DUMP_SDFG,
  DUMP_STD,
  DUMP_LLVM_DIALECT,
  DUMP_LLVM_IR,
  DUMP_OPTIMIZED_LLVM_IR,
  COMPILE,
};

namespace cmdline {
const std::string STDOUT = "-";
class OptionalSizeTParser : public llvm::cl::parser<llvm::Optional<size_t>> {
public:
  OptionalSizeTParser(llvm::cl::Option &option)
      : llvm::cl::parser<llvm::Optional<size_t>>(option) {}

  bool parse(llvm::cl::Option &option, llvm::StringRef argName,
             llvm::StringRef arg, llvm::Optional<size_t> &value) {
    size_t parsedVal;
    std::istringstream iss(arg.str());

    iss >> parsedVal;

    if (iss.fail())
      return option.error("Invalid value " + arg);

    value.emplace(parsedVal);

    return false;
  }
};

llvm::cl::list<std::string> inputs(llvm::cl::Positional,
                                   llvm::cl::desc("<Input files>"),
                                   llvm::cl::OneOrMore);

llvm::cl::opt<std::string> output("o",
                                  llvm::cl::desc("Specify output filename"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init(STDOUT));

llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("verbose logs"),
                            llvm::cl::init<bool>(false));

llvm::cl::opt<bool>
    optimizeTFHE("optimize-tfhe",
                 llvm::cl::desc("enable/disable optimizations of TFHE "
                                "dialects. (Enabled by default)"),
                 llvm::cl::init<bool>(true));

llvm::cl::opt<bool>
    simulate("simulate",
             llvm::cl::desc("enable/disable simulation of crypto operations "
                            "(Disabled by default)"),
             llvm::cl::init<bool>(false));

llvm::cl::opt<bool> emitGPUOps(
    "emit-gpu-ops",
    llvm::cl::desc(
        "enable/disable generating GPU operations (Disabled by default)"),
    llvm::cl::init<bool>(false));

llvm::cl::opt<bool> compressEvaluationKeys(
    "compress-inputs",
    llvm::cl::desc("Force the use of compressed (seeded) input "
                   "evaluation keys and ciphertexts"),
    llvm::cl::init<bool>(false));

llvm::cl::list<std::string> passes(
    "passes",
    llvm::cl::desc("Specify the passes to run (use only for compiler tests)"),
    llvm::cl::value_desc("passname"), llvm::cl::ZeroOrMore);

static llvm::cl::opt<enum Action> action(
    "a", "action", llvm::cl::desc("output mode"), llvm::cl::ValueRequired,
    llvm::cl::NumOccurrencesFlag::Required,
    llvm::cl::values(
        clEnumValN(Action::ROUND_TRIP, "roundtrip",
                   "Parse input module and regenerate textual representation")),
    llvm::cl::values(clEnumValN(Action::DUMP_FHE, "dump-fhe",
                                "Dump FHE module")),
    llvm::cl::values(clEnumValN(Action::DUMP_FHE_LINALG_GENERIC,
                                "dump-fhe-linalg-generic",
                                "Lower FHELinalg to Linalg and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_FHE_NO_LINALG,
                                "dump-fhe-no-linalg",
                                "Lower FHELinalg to FHE and dump result")),
    llvm::cl::values(
        clEnumValN(Action::DUMP_TFHE, "dump-tfhe",
                   "Lower to unparameterized TFHE and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_NORMALIZED_TFHE,
                                "dump-normalized-tfhe",
                                "Lower to normalized TFHE and dump result")),
    llvm::cl::values(clEnumValN(
        Action::DUMP_PARAMETRIZED_TFHE, "dump-parametrized-tfhe",
        "Lower to TFHE, parametrize TFHE operations and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_BATCHED_TFHE, "dump-batched-tfhe",
                                "Lower to TFHE, parametrize and then attempt "
                                "to batch TFHE operations")),
    llvm::cl::values(
        clEnumValN(Action::DUMP_SIMULATED_TFHE, "dump-simulated-tfhe",
                   "Lower to TFHE, then simulate crypto operations")),
    llvm::cl::values(clEnumValN(Action::DUMP_CONCRETE, "dump-concrete",
                                "Lower to Concrete and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_SDFG, "dump-sdfg",
                                "Lower to SDFG operations annd dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_STD, "dump-std",
                                "Lower to std and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_LLVM_DIALECT, "dump-llvm-dialect",
                                "Lower to LLVM dialect and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_LLVM_IR, "dump-llvm-ir",
                                "Lower to LLVM-IR and dump result")),
    llvm::cl::values(clEnumValN(Action::DUMP_OPTIMIZED_LLVM_IR,
                                "dump-optimized-llvm-ir",
                                "Lower to LLVM-IR, optimize and dump result")),
    llvm::cl::values(clEnumValN(Action::COMPILE, "compile",
                                "Lower to LLVM-IR, compile to a file")));

llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

llvm::cl::opt<bool> autoParallelize("parallelize",
                                    llvm::cl::desc("Generate parallel code"),
                                    llvm::cl::init(false));

llvm::cl::opt<bool> loopParallelize(
    "parallelize-loops",
    llvm::cl::desc("Generate parallel loops from Linalg operations"),
    llvm::cl::init(false));

llvm::cl::opt<bool> batchTFHEOps(
    "batch-tfhe-ops",
    llvm::cl::desc("Hoist scalar TFHE operations with corresponding batched "
                   "operations out of loop nests as batched operations"),
    llvm::cl::init(false));

llvm::cl::opt<int64_t>
    maxBatchSize("max-batch-size",
                 llvm::cl::desc("Maximum number of operands materialized in a "
                                "batch for --batch-tfhe-ops"),
                 llvm::cl::init(std::numeric_limits<int64_t>::max()));

llvm::cl::opt<bool> emitSDFGOps(
    "emit-sdfg-ops",
    llvm::cl::desc(
        "Extract operations supported by the SDFG dialect for static data flow"
        " graphs and emit them."),
    llvm::cl::init(false));

llvm::cl::opt<bool> unrollLoopsWithSDFGConvertibleOps(
    "unroll-loops-with-sdfg-convertible-ops",
    llvm::cl::desc("Causes loops containing SDFG-convertible operations to be "
                   "fully unrolled."),
    llvm::cl::init(false));

llvm::cl::opt<bool> dataflowParallelize(
    "parallelize-dataflow",
    llvm::cl::desc("Generate the program as a dataflow graph"),
    llvm::cl::init(false));

llvm::cl::opt<std::string>
    funcName("funcname",
             llvm::cl::desc("Name of the function to compile, default 'main'"),
             llvm::cl::init<std::string>(""));

llvm::cl::opt<bool>
    chunkIntegers("chunk-integers",
                  llvm::cl::desc("Whether to decompose integer into chunks or "
                                 "not, default is false (to not chunk)"),
                  llvm::cl::init<bool>(false));

llvm::cl::opt<unsigned int> chunkSize(
    "chunk-size",
    llvm::cl::desc(
        "Chunk size while decomposing big integers into chunks, default is 4"),
    llvm::cl::init<unsigned int>(4));

llvm::cl::opt<unsigned int> chunkWidth(
    "chunk-width",
    llvm::cl::desc(
        "Chunk width while decomposing big integers into chunks, default is 2"),
    llvm::cl::init<unsigned int>(2));

llvm::cl::opt<double> pbsErrorProbability(
    "pbs-error-probability",
    llvm::cl::desc("Change the default probability of error for all pbs"),
    llvm::cl::init(optimizer::DEFAULT_CONFIG.p_error));

llvm::cl::opt<double> globalErrorProbability(
    "global-error-probability",
    llvm::cl::desc(
        "Use global error probability (override pbs error probability)"),
    llvm::cl::init(optimizer::DEFAULT_CONFIG.global_p_error));

llvm::cl::opt<double> securityLevel(
    "security-level",
    llvm::cl::desc(
        "Specify the security level to target for compiling the program"),
    llvm::cl::init(optimizer::DEFAULT_CONFIG.security));

llvm::cl::opt<bool> displayOptimizerChoice(
    "display-optimizer-choice",
    llvm::cl::desc("Display the information returned by the optimizer"),
    llvm::cl::init(false));

llvm::cl::opt<optimizer::Strategy> optimizerStrategy(
    "optimizer-strategy",
    llvm::cl::desc("Select the concrete optimizer strategy"),
    llvm::cl::init(optimizer::DEFAULT_STRATEGY),
    llvm::cl::values(clEnumValN(optimizer::Strategy::V0, "V0",
                                "Use the V0 optimizer strategy that use the "
                                "worst case atomic pattern")),
    llvm::cl::values(clEnumValN(
        optimizer::Strategy::DAG_MONO, "dag-mono",
        "Use the dag-mono optimizer strategy that solve the optimization "
        "problem using the fhe computation dag with ONE set of evaluation "
        "keys")),
    llvm::cl::values(clEnumValN(
        optimizer::Strategy::DAG_MULTI, "dag-multi",
        "Use the dag-multi optimizer strategy that solve the optimization "
        "problem using the fhe computation dag with SEVERAL set of evaluation "
        "keys")));

llvm::cl::opt<bool> optimizerKeySharing(
    "optimizer-multi-parameter-key-sharing",
    llvm::cl::desc(
        "To enable/disable key sharing in dag-multi parameter strategy"),
    llvm::cl::init(optimizer::DEFAULT_KEY_SHARING));

llvm::cl::opt<double> fallbackLogNormWoppbs(
    "optimizer-fallback-log-norm-woppbs",
    llvm::cl::desc("Select a fallback value for multisum log norm in woppbs "
                   "when the precise value can't be computed."),
    llvm::cl::init(optimizer::DEFAULT_CONFIG.fallback_log_norm_woppbs));

llvm::cl::opt<concrete_optimizer::MultiParamStrategy>
    optimizerMultiParamStrategy(
        "optimizer-multi-parameter-strategy",
        llvm::cl::desc(
            "Select the concrete optimizer multi parameter strategy"),
        llvm::cl::init(optimizer::DEFAULT_MULTI_PARAM_STRATEGY),
        llvm::cl::values(clEnumValN(
            concrete_optimizer::MultiParamStrategy::ByPrecision, "by-precision",
            "One partition set for each possible input TLU precision")),
        llvm::cl::values(clEnumValN(
            concrete_optimizer::MultiParamStrategy::ByPrecisionAndNorm2,
            "by-precision-and-norm2",
            "One partition set for each possible input TLU precision and "
            "output norm2")));

llvm::cl::opt<concrete_optimizer::Encoding> optimizerEncoding(
    "force-encoding", llvm::cl::desc("Choose cyphertext encoding."),
    llvm::cl::init(optimizer::DEFAULT_CONFIG.encoding),
    llvm::cl::values(clEnumValN(concrete_optimizer::Encoding::Auto, "auto",
                                "Pick the best [default]")),
    llvm::cl::values(clEnumValN(concrete_optimizer::Encoding::Native, "native",
                                "native")),
    llvm::cl::values(clEnumValN(concrete_optimizer::Encoding::Crt, "crt",
                                "Chineese Reminder Theorem representation")));

llvm::cl::opt<bool> optimizerNoCacheOnDisk(
    "optimizer-no-cache-on-disk",
    llvm::cl::desc("Optimizer cache is sync from/to disk. Usefull to debug "
                   "cache issues."),
    llvm::cl::init(false));

llvm::cl::opt<bool> optimizerAllowComposition(
    "optimizer-allow-composition",
    llvm::cl::desc("Optimizer is parameterized to allow calling the circuit on "
                   "its own output without decryptions."),
    llvm::cl::init(false));

llvm::cl::list<int64_t> fhelinalgTileSizes(
    "fhelinalg-tile-sizes",
    llvm::cl::desc(
        "Force tiling of FHELinalg operation with the given tile sizes"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<size_t> v0Constraint(
    "v0-constraint",
    llvm::cl::desc(
        "Force the compiler to use the given v0 constraint [p, norm2]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<size_t> v0Parameter(
    "v0-parameter",
    llvm::cl::desc(
        "Force to apply the given v0 parameters [glweDimension, "
        "logPolynomialSize, nSmall, brLevel, brLobBase, ksLevel, ksLogBase]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<int64_t> largeIntegerCRTDecomposition(
    "large-integer-crt-decomposition",
    llvm::cl::desc(
        "Use the large integer to lower FHE.eint with the given decomposition, "
        "must be used with the other large-integers options (experimental)"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<int64_t> largeIntegerPackingKeyswitch(
    "large-integer-packing-keyswitch",
    llvm::cl::desc(
        "Use the large integer to lower FHE.eint with the given parameters for "
        "packing keyswitch, must be used with the other large-integers options "
        "(experimental) [inputLweDimension, outputPolynomialSize, level, "
        "baseLog]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::list<int64_t> largeIntegerCircuitBootstrap(
    "large-integer-circuit-bootstrap",
    llvm::cl::desc(
        "Use the large integer to lower FHE.eint with the given parameters for "
        "the cicuit boostrap, must be used with the other large-integers "
        "options "
        "(experimental) [level, baseLog]"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

llvm::cl::opt<std::string> circuitEncodings(
    "circuit-encodings",
    llvm::cl::desc("Specify the input and output encodings of the circuit, "
                   "using the JSON representation."),
    llvm::cl::init(std::string{}));

} // namespace cmdline

namespace llvm {
// This needs to be wrapped into the llvm namespace for proper
// operator lookup
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const llvm::ArrayRef<uint64_t> arr) {
  os << "(";
  for (size_t i = 0; i < arr.size(); i++) {
    os << arr[i];

    if (i != arr.size() - 1)
      os << ", ";
  }

  return os;
}
} // namespace llvm

llvm::Expected<mlir::concretelang::CompilationOptions>
cmdlineCompilationOptions() {
  mlir::concretelang::CompilationOptions options;

  options.verifyDiagnostics = cmdline::verifyDiagnostics;
  options.autoParallelize = cmdline::autoParallelize;
  options.loopParallelize = cmdline::loopParallelize;
  options.dataflowParallelize = cmdline::dataflowParallelize;
  options.batchTFHEOps = cmdline::batchTFHEOps;
  options.maxBatchSize = cmdline::maxBatchSize;
  options.emitSDFGOps = cmdline::emitSDFGOps;
  options.unrollLoopsWithSDFGConvertibleOps =
      cmdline::unrollLoopsWithSDFGConvertibleOps;
  options.optimizeTFHE = cmdline::optimizeTFHE;
  options.simulate = cmdline::simulate;
  options.emitGPUOps = cmdline::emitGPUOps;
  options.compressEvaluationKeys = cmdline::compressEvaluationKeys;
  options.chunkIntegers = cmdline::chunkIntegers;
  options.chunkSize = cmdline::chunkSize;
  options.chunkWidth = cmdline::chunkWidth;

  if (!cmdline::v0Constraint.empty()) {
    if (cmdline::v0Constraint.size() != 2) {
      return llvm::make_error<llvm::StringError>(
          "The v0-constraint option expect a list of size 2",
          llvm::inconvertibleErrorCode());
    }
    options.v0FHEConstraints = mlir::concretelang::V0FHEConstraint{
        cmdline::v0Constraint[1], cmdline::v0Constraint[0]};
  }

  if (!cmdline::funcName.empty()) {
    options.mainFuncName = cmdline::funcName;
  }

  // Convert tile sizes to `Optional`
  if (!cmdline::fhelinalgTileSizes.empty())
    options.fhelinalgTileSizes.emplace(cmdline::fhelinalgTileSizes);

  // Setup the v0 parameter options
  if (!cmdline::v0Parameter.empty()) {
    if (cmdline::v0Parameter.size() != 7) {
      return llvm::make_error<llvm::StringError>(
          "The v0-parameter option expect a list of size 7",
          llvm::inconvertibleErrorCode());
    }
    options.v0Parameter = {cmdline::v0Parameter[0], cmdline::v0Parameter[1],
                           cmdline::v0Parameter[2], cmdline::v0Parameter[3],
                           cmdline::v0Parameter[4], cmdline::v0Parameter[5],
                           cmdline::v0Parameter[6], std::nullopt};
  }

  // Setup the large integer options
  if (!cmdline::largeIntegerCRTDecomposition.empty() ||
      !cmdline::largeIntegerPackingKeyswitch.empty() ||
      !cmdline::largeIntegerPackingKeyswitch.empty()) {
    if (cmdline::largeIntegerCRTDecomposition.empty() ||
        cmdline::largeIntegerPackingKeyswitch.empty() ||
        cmdline::largeIntegerPackingKeyswitch.empty()) {
      return llvm::make_error<llvm::StringError>(
          "The large-integers options should all be set",
          llvm::inconvertibleErrorCode());
    }
    if (cmdline::largeIntegerPackingKeyswitch.size() != 4) {
      return llvm::make_error<llvm::StringError>(
          "The large-integers-packing-keyswitch must be a list of 4 integer",
          llvm::inconvertibleErrorCode());
    }
    if (cmdline::largeIntegerCircuitBootstrap.size() != 2) {
      return llvm::make_error<llvm::StringError>(
          "The large-integers-packing-keyswitch must be a list of 2 integer",
          llvm::inconvertibleErrorCode());
    }
    options.largeIntegerParameter = mlir::concretelang::LargeIntegerParameter();
    options.largeIntegerParameter->crtDecomposition =
        cmdline::largeIntegerCRTDecomposition;
    options.largeIntegerParameter->wopPBS.packingKeySwitch.inputLweDimension =
        cmdline::largeIntegerPackingKeyswitch[0];
    options.largeIntegerParameter->wopPBS.packingKeySwitch
        .outputPolynomialSize = cmdline::largeIntegerPackingKeyswitch[1];
    options.largeIntegerParameter->wopPBS.packingKeySwitch.level =
        cmdline::largeIntegerPackingKeyswitch[2];
    options.largeIntegerParameter->wopPBS.packingKeySwitch.baseLog =
        cmdline::largeIntegerPackingKeyswitch[3];
    options.largeIntegerParameter->wopPBS.circuitBootstrap.level =
        cmdline::largeIntegerCircuitBootstrap[0];
    options.largeIntegerParameter->wopPBS.circuitBootstrap.baseLog =
        cmdline::largeIntegerCircuitBootstrap[1];
  }

  options.optimizerConfig.global_p_error = cmdline::globalErrorProbability;
  options.optimizerConfig.p_error = cmdline::pbsErrorProbability;
  options.optimizerConfig.display = cmdline::displayOptimizerChoice;
  options.optimizerConfig.strategy = cmdline::optimizerStrategy;
  options.optimizerConfig.key_sharing = cmdline::optimizerKeySharing;
  options.optimizerConfig.multi_param_strategy =
      cmdline::optimizerMultiParamStrategy;
  options.optimizerConfig.encoding = cmdline::optimizerEncoding;
  options.optimizerConfig.cache_on_disk = !cmdline::optimizerNoCacheOnDisk;
  options.optimizerConfig.composable = cmdline::optimizerAllowComposition;

  if (!std::isnan(options.optimizerConfig.global_p_error) &&
      options.optimizerConfig.strategy == optimizer::Strategy::V0) {
    return llvm::make_error<llvm::StringError>(
        "--global-error-probability is not compatible with --optimizer-v0",
        llvm::inconvertibleErrorCode());
  }

  if (!cmdline::circuitEncodings.empty()) {
    auto jsonString = cmdline::circuitEncodings.getValue();
    auto encodings = Message<concreteprotocol::CircuitEncodingInfo>();
    if (encodings.readJsonFromString(jsonString).has_failure()) {
      return llvm::make_error<llvm::StringError>(
          "Failed to parse the --circuit-encodings option",
          llvm::inconvertibleErrorCode());
    }
    options.encodings = encodings;
  }

  return options;
}

/// Process a single source buffer
///
/// The parameter `action` specifies how the buffer should be processed
/// and thus defines the output.
///
/// The parameter `parametrizeTFHE` defines, whether the
/// parametrization pass for TFHE is executed. If the `action` does
/// not involve any MidlFHE manipulation, this parameter does not have
/// any effect.
///
/// The parameters `overrideMaxEintPrecision` and `overrideMaxMANP`, if
/// set, override the values for the maximum required precision of
/// encrypted integers and the maximum value for the Minimum Arithmetic
/// Noise Padding otherwise determined automatically.
///
/// If `verifyDiagnostics` is `true`, the procedure only checks if the
/// diagnostic messages provided in the source buffer using
/// `expected-error` are produced. If `verifyDiagnostics` is `false`,
/// the procedure checks if the parsed module is valid and if all
/// requested transformations succeeded.
///
/// Compilation output is written to the stream specified by `os`.
mlir::LogicalResult processInputBuffer(
    std::unique_ptr<llvm::MemoryBuffer> buffer, std::string sourceFileName,
    mlir::concretelang::CompilationOptions &options, enum Action action,
    llvm::raw_ostream &os,
    std::shared_ptr<mlir::concretelang::CompilerEngine::Library> outputLib) {
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();

  std::string funcName = options.mainFuncName.value_or("");

  mlir::concretelang::CompilerEngine ce{ccx};
  ce.setCompilationOptions(std::move(options));

  if (cmdline::passes.size() != 0) {
    ce.setEnablePass([](mlir::Pass *pass) {
      return std::any_of(
          cmdline::passes.begin(), cmdline::passes.end(),
          [&](const std::string &p) { return pass->getArgument() == p; });
    });
  }
  enum mlir::concretelang::CompilerEngine::Target target;

  switch (action) {
  case Action::ROUND_TRIP:
    target = mlir::concretelang::CompilerEngine::Target::ROUND_TRIP;
    break;
  case Action::DUMP_FHE:
    target = mlir::concretelang::CompilerEngine::Target::FHE;
    break;
  case Action::DUMP_FHE_LINALG_GENERIC:
    target = mlir::concretelang::CompilerEngine::Target::FHE_LINALG_GENERIC;
    break;
  case Action::DUMP_FHE_NO_LINALG:
    target = mlir::concretelang::CompilerEngine::Target::FHE_NO_LINALG;
    break;
  case Action::DUMP_TFHE:
    target = mlir::concretelang::CompilerEngine::Target::TFHE;
    break;
  case Action::DUMP_NORMALIZED_TFHE:
    target = mlir::concretelang::CompilerEngine::Target::NORMALIZED_TFHE;
    break;
  case Action::DUMP_PARAMETRIZED_TFHE:
    target = mlir::concretelang::CompilerEngine::Target::PARAMETRIZED_TFHE;
    break;
  case Action::DUMP_BATCHED_TFHE:
    target = mlir::concretelang::CompilerEngine::Target::BATCHED_TFHE;
    break;
  case Action::DUMP_SIMULATED_TFHE:
    target = mlir::concretelang::CompilerEngine::Target::SIMULATED_TFHE;
    break;
  case Action::DUMP_CONCRETE:
    target = mlir::concretelang::CompilerEngine::Target::CONCRETE;
    break;
  case Action::DUMP_SDFG:
    target = mlir::concretelang::CompilerEngine::Target::SDFG;
    break;
  case Action::DUMP_STD:
    target = mlir::concretelang::CompilerEngine::Target::STD;
    break;
  case Action::DUMP_LLVM_DIALECT:
    target = mlir::concretelang::CompilerEngine::Target::LLVM;
    break;
  case Action::DUMP_LLVM_IR:
    target = mlir::concretelang::CompilerEngine::Target::LLVM_IR;
    break;
  case Action::DUMP_OPTIMIZED_LLVM_IR:
    target = mlir::concretelang::CompilerEngine::Target::OPTIMIZED_LLVM_IR;
    break;
  case Action::COMPILE:
    target = mlir::concretelang::CompilerEngine::Target::LIBRARY;
    break;
  }
  auto retOrErr = ce.compile(std::move(buffer), target, outputLib);

  if (!retOrErr) {
    mlir::concretelang::log_error()
        << llvm::toString(retOrErr.takeError()) << "\n";

    return mlir::failure();
  }

  if (retOrErr->llvmModule) {
    // At least usefull for intermediate binary object files naming
    retOrErr->llvmModule->setSourceFileName(sourceFileName);
    retOrErr->llvmModule->setModuleIdentifier(sourceFileName);
  }

  if (options.verifyDiagnostics) {
    return mlir::success();
  } else if (action == Action::DUMP_LLVM_IR ||
             action == Action::DUMP_OPTIMIZED_LLVM_IR) {
    retOrErr->llvmModule->print(os, nullptr);
  } else if (action != Action::COMPILE) {
    retOrErr->mlirModuleRef->get().print(os);
  }

  return mlir::success();
}

mlir::LogicalResult compilerMain(int argc, char **argv) {
  // Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::concretelang::setupLogging(cmdline::verbose);

  // String for error messages
  std::string errorMessage;

  if (cmdline::action == Action::COMPILE) {
    if (cmdline::output == cmdline::STDOUT) {
      // can't use stdin to generate a lib.
      errorMessage += "Please provide a file destination '-o' option.\n";
    }
    // SplitInputFile would need to have separate object files
    // destinations to be able to work.
    if (cmdline::splitInputFile) {
      errorMessage +=
          "'--action=compile' and '--split-input-file' are incompatible\n";
    }
    if (errorMessage != "") {
      llvm::errs() << errorMessage << "\n";
      return mlir::failure();
    }
  }

  auto compilerOptions = cmdlineCompilationOptions();
  if (auto err = compilerOptions.takeError()) {
    llvm::errs() << err << "\n";
    return mlir::failure();
  }

  // In case of compilation to library, the real output is the library.
  std::string outputPath =
      (cmdline::action == Action::COMPILE) ? cmdline::STDOUT : cmdline::output;

  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputPath, &errorMessage);

  using Library = mlir::concretelang::CompilerEngine::Library;
  auto outputLib = std::make_shared<Library>(cmdline::output);

  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  // Iterate over all input files specified on the command line
  for (const auto &fileName : cmdline::inputs) {
    auto file = mlir::openInputFile(fileName, &errorMessage);

    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return mlir::failure();
    }

    // If `--split-input-file` is set, the file is split into
    // individual chunks separated by `// -----` markers. Each chunk
    // is then processed individually as if it were part of a separate
    // source file.
    auto process = [&](std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                       llvm::raw_ostream &os) {
      return processInputBuffer(std::move(inputBuffer), fileName,
                                *compilerOptions, cmdline::action, os,
                                outputLib);
    };
    auto &os = output->os();
    auto res = mlir::failure();
    if (cmdline::splitInputFile) {
      res = mlir::splitAndProcessBuffer(std::move(file), process, os);
    } else {
      res = process(std::move(file), os);
    }
    if (res.failed()) {
      return mlir::failure();
    } else {
      output->keep();
    }
  }

  if (cmdline::action == Action::COMPILE) {
    auto err = outputLib->emitArtifacts(
        /*sharedLib=*/true, /*staticLib=*/true,
        /*clientParameters=*/true, /*compilationFeedback=*/true);
    if (err) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  int result = 0;
  if (mlir::failed(compilerMain(argc, argv)))
    result = 1;

  _dfr_terminate();
  return result;
}
