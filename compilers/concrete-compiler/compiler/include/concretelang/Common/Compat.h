// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.
//
// NOTE:
// -----
// To limit the size of the refactoring, we chose to not propagate the new
// client/server lib to the exterior APIs after it was finalized. This file only
// serves as a compatibility layer for exterior (python/rust/c) apis, for the
// time being.

#ifndef CONCRETELANG_COMMON_COMPAT
#define CONCRETELANG_COMMON_COMPAT

#include "capnp/serialize-packed.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Keys.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Values.h"
#include "concretelang/ServerLib/ServerLib.h"
#include "kj/io.h"
#include "kj/std/iostream.h"
#include <concretelang/ClientLib/ClientLib.h>
#include <concretelang/Common/Keysets.h>
#include <concretelang/Common/Values.h>
#include <concretelang/ServerLib/ServerLib.h>
#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Error.h>
#include <memory>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <stdexcept>

using concretelang::clientlib::ClientCircuit;
using concretelang::clientlib::ClientProgram;
using concretelang::keysets::Keyset;
using concretelang::keysets::KeysetCache;
using concretelang::keysets::ServerKeyset;
using concretelang::serverlib::ServerCircuit;
using concretelang::serverlib::ServerProgram;
using concretelang::values::TransportValue;
using concretelang::values::Value;

#define GET_OR_THROW_LLVM_EXPECTED(VARNAME, EXPECTED)                          \
  auto VARNAME = EXPECTED;                                                     \
  if (auto err = VARNAME.takeError()) {                                        \
    throw std::runtime_error(llvm::toString(std::move(err)));                  \
  }

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define GET_OR_THROW_RESULT_(VARNAME, RESULT, MAYBE)                           \
  auto MAYBE = RESULT;                                                         \
  if (MAYBE.has_failure()) {                                                   \
    throw std::runtime_error(MAYBE.as_failure().error().mesg);                 \
  }                                                                            \
  VARNAME = MAYBE.value();

#define GET_OR_THROW_RESULT(VARNAME, RESULT)                                   \
  GET_OR_THROW_RESULT_(VARNAME, RESULT, CONCAT(maybe, __COUNTER__))

#define EXPECTED_TRY_(lhs, rhs, maybe)                                         \
  auto maybe = rhs;                                                            \
  if (auto err = maybe.takeError()) {                                          \
    return std::move(err);                                                     \
  }                                                                            \
  lhs = *maybe;

#define EXPECTED_TRY(lhs, rhs)                                                 \
  EXPECTED_TRY_(lhs, rhs, CONCAT(maybe, __COUNTER__))

template <typename T> llvm::Expected<T> outcomeToExpected(Result<T> outcome) {
  if (outcome.has_failure()) {
    return mlir::concretelang::StreamStringError(
        outcome.as_failure().error().mesg);
  } else {
    return outcome.value();
  }
}

// Every number sent by python through the API has a type `int64` that must be
// turned into the proper type expected by the ArgTransformers. This allows to
// get an extra transformer executed right before the ArgTransformer gets
// called.
std::function<Value(Value)>
getPythonTypeTransformer(const Message<concreteprotocol::GateInfo> &info) {
  if (info.asReader().getTypeInfo().hasIndex()) {
    return [=](Value input) {
      Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
      return Value{(Tensor<uint64_t>)tensorInput};
    };
  } else if (info.asReader().getTypeInfo().hasPlaintext()) {
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        8) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint8_t>)tensorInput};
      };
    }
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        16) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint16_t>)tensorInput};
      };
    }
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        32) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint32_t>)tensorInput};
      };
    }
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        64) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint64_t>)tensorInput};
      };
    }
    assert(false);
  } else if (info.asReader().getTypeInfo().hasLweCiphertext()) {
    if (info.asReader()
            .getTypeInfo()
            .getLweCiphertext()
            .getEncoding()
            .hasInteger() &&
        info.asReader()
            .getTypeInfo()
            .getLweCiphertext()
            .getEncoding()
            .getInteger()
            .getIsSigned()) {
      return [=](Value input) { return input; };
    } else {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint64_t>)tensorInput};
      };
    }
  } else {
    assert(false);
  }
};

namespace concretelang {
namespace serverlib {
/// A transition structure that preserver the current API of the library
/// support.
struct ServerLambda {
  ServerCircuit circuit;
  bool isSimulation;
};
} // namespace serverlib

namespace clientlib {

/// A transition structure that preserver the current API of the library
/// support.
struct LweSecretKeyParam {
  Message<concreteprotocol::LweSecretKeyInfo> info;
};

/// A transition structure that preserver the current API of the library
/// support.
struct BootstrapKeyParam {
  Message<concreteprotocol::LweBootstrapKeyInfo> info;
};

/// A transition structure that preserver the current API of the library
/// support.
struct KeyswitchKeyParam {
  Message<concreteprotocol::LweKeyswitchKeyInfo> info;
};

/// A transition structure that preserver the current API of the library
/// support.
struct PackingKeyswitchKeyParam {
  Message<concreteprotocol::PackingKeyswitchKeyInfo> info;
};

/// A transition structure that preserver the current API of the library
/// support.
struct Encoding {
  Message<concreteprotocol::EncodingInfo> circuit;
};

/// A transition structure that preserver the current API of the library
/// support.
struct EncryptionGate {
  Message<concreteprotocol::GateInfo> gateInfo;
};

/// A transition structure that preserver the current API of the library
/// support.
struct CircuitGate {
  Message<concreteprotocol::GateInfo> gateInfo;
};

/// A transition structure that preserver the current API of the library
/// support.
struct ValueExporter {
  ClientCircuit circuit;
};

/// A transition structure that preserver the current API of the library
/// support.
struct SimulatedValueExporter {
  ClientCircuit circuit;
};

/// A transition structure that preserver the current API of the library
/// support.
struct ValueDecrypter {
  ClientCircuit circuit;
};

/// A transition structure that preserver the current API of the library
/// support.
struct SimulatedValueDecrypter {
  ClientCircuit circuit;
};

/// A transition structure that preserver the current API of the library
/// support.
struct PublicArguments {
  std::vector<TransportValue> values;
};

/// A transition structure that preserver the current API of the library
/// support.
struct PublicResult {
  std::vector<TransportValue> values;
};

/// A transition structure that preserver the current API of the library
/// support.
struct SharedScalarOrTensorData {
  TransportValue value;
};

/// A transition structure that preserver the current API of the library
/// support.
struct ClientParameters {
  Message<concreteprotocol::ProgramInfo> programInfo;
  std::vector<LweSecretKeyParam> secretKeys;
  std::vector<BootstrapKeyParam> bootstrapKeys;
  std::vector<KeyswitchKeyParam> keyswitchKeys;
  std::vector<PackingKeyswitchKeyParam> packingKeyswitchKeys;
};

/// A transition structure that preserver the current API of the library
/// support.
struct EvaluationKeys {
  ServerKeyset keyset;
};

/// A transition structure that preserver the current API of the library
/// support.
struct KeySetCache {
  KeysetCache keysetCache;
};

/// A transition structure that preserver the current API of the library
/// support.
struct KeySet {
  Keyset keyset;
  Message<concreteprotocol::ProgramInfo> programInfo;
};

// integers are not serialized as binary values even on a binary stream
// so we cannot rely on << operator directly
template <typename Word>
std::ostream &writeWord(std::ostream &ostream, Word word) {
  ostream.write(reinterpret_cast<char *>(&(word)), sizeof(word));
  assert(ostream.good());
  return ostream;
}

template <typename Size>
std::ostream &writeSize(std::ostream &ostream, Size size) {
  return writeWord(ostream, size);
}

// for sake of symetry
template <typename Word>
std::istream &readWord(std::istream &istream, Word &word) {
  istream.read(reinterpret_cast<char *>(&(word)), sizeof(word));
  assert(istream.good());
  return istream;
}

template <typename Word>
std::istream &readWords(std::istream &istream, Word *words, size_t numWords) {
  assert(std::numeric_limits<size_t>::max() / sizeof(*words) > numWords);
  istream.read(reinterpret_cast<char *>(words), sizeof(*words) * numWords);
  assert(istream.good());
  return istream;
}

template <typename Size>
std::istream &readSize(std::istream &istream, Size &size) {
  return readWord(istream, size);
}

std::ostream &writeUInt64KeyBuffer(std::ostream &ostream,
                                   std::vector<uint64_t> &buffer) {
  writeSize(ostream, (uint64_t)buffer.size());
  ostream.write((const char *)buffer.data(), buffer.size() * sizeof(uint64_t));
  assert(ostream.good());
  return ostream;
}

std::istream &operator>>(std::istream &istream,
                         std::shared_ptr<std::vector<uint64_t>> &vec) {
  // TODO assertion on size?
  uint64_t size;
  readSize(istream, size);
  vec->resize(size);
  istream.read((char *)vec->data(), size * sizeof(uint64_t));
  assert(istream.good());
  return istream;
}

// LweSecretKey /////////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweSecretKey &key) {
  assert(key.toProto().writeBinaryToOstream(ostream).has_value());
  return ostream;
}

LweSecretKey readLweSecretKey(std::istream &istream) {
  auto keyProto = Message<concreteprotocol::LweSecretKey>();
  assert(keyProto.readBinaryFromIstream(istream).has_value());
  return LweSecretKey::fromProto(keyProto);
}

// LweKeyswitchKey //////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweKeyswitchKey &key) {
  assert(key.toProto().writeBinaryToOstream(ostream).has_value());
  return ostream;
}

LweKeyswitchKey readLweKeyswitchKey(std::istream &istream) {
  auto keyProto = Message<concreteprotocol::LweKeyswitchKey>();
  assert(keyProto.readBinaryFromIstream(istream).has_value());
  return LweKeyswitchKey::fromProto(keyProto);
}

// LweBootstrapKey //////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweBootstrapKey &key) {
  assert(key.toProto().writeBinaryToOstream(ostream).has_value());
  return ostream;
}

LweBootstrapKey readLweBootstrapKey(std::istream &istream) {
  auto keyProto = Message<concreteprotocol::LweBootstrapKey>();
  assert(keyProto.readBinaryFromIstream(istream).has_value());
  return LweBootstrapKey::fromProto(keyProto);
}

// PackingKeyswitchKey //////////////////////////////

std::ostream &operator<<(std::ostream &ostream,
                         const PackingKeyswitchKey &key) {
  assert(key.toProto().writeBinaryToOstream(ostream).has_value());
  return ostream;
}

PackingKeyswitchKey readPackingKeyswitchKey(std::istream &istream) {
  auto keyProto = Message<concreteprotocol::PackingKeyswitchKey>();
  assert(keyProto.readBinaryFromIstream(istream).has_value());
  return PackingKeyswitchKey::fromProto(keyProto);
}

// KeySet ////////////////////////////////

std::unique_ptr<KeySet> readKeySet(std::istream &istream) {
  uint64_t nbKey;

  readSize(istream, nbKey);
  std::vector<LweSecretKey> secretKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    secretKeys.push_back(readLweSecretKey(istream));
  }

  readSize(istream, nbKey);
  std::vector<LweBootstrapKey> bootstrapKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    bootstrapKeys.push_back(readLweBootstrapKey(istream));
  }

  readSize(istream, nbKey);
  std::vector<LweKeyswitchKey> keyswitchKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    keyswitchKeys.push_back(readLweKeyswitchKey(istream));
  }

  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
  readSize(istream, nbKey);
  for (uint64_t i = 0; i < nbKey; i++) {
    packingKeyswitchKeys.push_back(readPackingKeyswitchKey(istream));
  }

  auto programInfo = Message<concreteprotocol::ProgramInfo>();
  assert(programInfo.readBinaryFromIstream(istream).has_value());

  auto clientKeyset = keysets::ClientKeyset{secretKeys};
  auto serverKeyset =
      keysets::ServerKeyset{bootstrapKeys, keyswitchKeys, packingKeyswitchKeys};
  auto keyset = keysets::Keyset{serverKeyset, clientKeyset};

  return std::make_unique<KeySet>(KeySet{keyset, programInfo});
}

std::ostream &operator<<(std::ostream &ostream, const KeySet &keySet) {
  auto secretKeys = keySet.keyset.client.lweSecretKeys;
  writeSize(ostream, secretKeys.size());
  for (auto sk : secretKeys) {
    ostream << sk;
  }

  auto bootstrapKeys = keySet.keyset.server.lweBootstrapKeys;
  writeSize(ostream, bootstrapKeys.size());
  for (auto bsk : bootstrapKeys) {
    ostream << bsk;
  }

  auto keyswitchKeys = keySet.keyset.server.lweKeyswitchKeys;
  writeSize(ostream, keyswitchKeys.size());
  for (auto ksk : keyswitchKeys) {
    ostream << ksk;
  }

  auto packingKeyswitchKeys = keySet.keyset.server.packingKeyswitchKeys;
  writeSize(ostream, packingKeyswitchKeys.size());
  for (auto pksk : packingKeyswitchKeys) {
    ostream << pksk;
  }

  assert(keySet.programInfo.writeBinaryToOstream(ostream).has_value());

  assert(ostream.good());
  return ostream;
}

// EvaluationKey ////////////////////////////////

EvaluationKeys readEvaluationKeys(std::istream &istream) {
  uint64_t nbKey;
  readSize(istream, nbKey);
  std::vector<LweBootstrapKey> bootstrapKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    bootstrapKeys.push_back(readLweBootstrapKey(istream));
  }
  readSize(istream, nbKey);
  std::vector<LweKeyswitchKey> keyswitchKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    keyswitchKeys.push_back(readLweKeyswitchKey(istream));
  }
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
  readSize(istream, nbKey);
  for (uint64_t i = 0; i < nbKey; i++) {
    packingKeyswitchKeys.push_back(readPackingKeyswitchKey(istream));
  }
  auto serverKeyset =
      keysets::ServerKeyset{bootstrapKeys, keyswitchKeys, packingKeyswitchKeys};
  return EvaluationKeys{serverKeyset};
}

std::ostream &operator<<(std::ostream &ostream,
                         const EvaluationKeys &evaluationKeys) {
  auto bootstrapKeys = evaluationKeys.keyset.lweBootstrapKeys;
  writeSize(ostream, bootstrapKeys.size());
  for (auto bsk : bootstrapKeys) {
    ostream << bsk;
  }
  auto keyswitchKeys = evaluationKeys.keyset.lweKeyswitchKeys;
  writeSize(ostream, keyswitchKeys.size());
  for (auto ksk : keyswitchKeys) {
    ostream << ksk;
  }
  auto packingKeyswitchKeys = evaluationKeys.keyset.packingKeyswitchKeys;
  writeSize(ostream, packingKeyswitchKeys.size());
  for (auto pksk : packingKeyswitchKeys) {
    ostream << pksk;
  }
  assert(ostream.good());
  return ostream;
}

} // namespace clientlib
} // namespace concretelang

namespace mlir {
namespace concretelang {

/// A transition structure that preserver the current API of the library
/// support.
struct LambdaArgument {
  ::concretelang::values::Value value;
};

/// LibraryCompilationResult is the result of a compilation to a library.
struct LibraryCompilationResult {
  /// The output directory path where the compilation artifacts have been
  /// generated.
  std::string outputDirPath;
  std::string funcName;
};

class LibrarySupport {

public:
  LibrarySupport(std::string outputPath, std::string runtimeLibraryPath = "",
                 bool generateSharedLib = true, bool generateStaticLib = true,
                 bool generateClientParameters = true,
                 bool generateCompilationFeedback = true)
      : outputPath(outputPath), runtimeLibraryPath(runtimeLibraryPath),
        generateSharedLib(generateSharedLib),
        generateStaticLib(generateStaticLib),
        generateClientParameters(generateClientParameters),
        generateCompilationFeedback(generateCompilationFeedback) {}

  llvm::Expected<std::unique_ptr<LibraryCompilationResult>>
  compile(llvm::SourceMgr &program, CompilationOptions options) {
    // Setup the compiler engine
    auto context = CompilationContext::createShared();
    concretelang::CompilerEngine engine(context);
    engine.setCompilationOptions(options);

    // Compile to a library
    auto library =
        engine.compile(program, outputPath, runtimeLibraryPath,
                       generateSharedLib, generateStaticLib,
                       generateClientParameters, generateCompilationFeedback);
    if (auto err = library.takeError()) {
      return std::move(err);
    }

    if (!options.mainFuncName.has_value()) {
      return StreamStringError("Need to have a funcname to compile library");
    }
    this->funcName = options.mainFuncName.value();

    auto result = std::make_unique<LibraryCompilationResult>();
    result->outputDirPath = outputPath;
    result->funcName = *options.mainFuncName;
    return std::move(result);
  }

  llvm::Expected<std::unique_ptr<LibraryCompilationResult>>
  compile(llvm::StringRef s, CompilationOptions options) {
    std::unique_ptr<llvm::MemoryBuffer> mb =
        llvm::MemoryBuffer::getMemBuffer(s);
    llvm::SourceMgr sm;
    sm.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
    return this->compile(sm, options);
  }

  llvm::Expected<std::unique_ptr<LibraryCompilationResult>>
  compile(mlir::ModuleOp &program,
          std::shared_ptr<mlir::concretelang::CompilationContext> &context,
          CompilationOptions options) {

    // Setup the compiler engine
    concretelang::CompilerEngine engine(context);
    engine.setCompilationOptions(options);

    // Compile to a library
    auto library =
        engine.compile(program, outputPath, runtimeLibraryPath,
                       generateSharedLib, generateStaticLib,
                       generateClientParameters, generateCompilationFeedback);
    if (auto err = library.takeError()) {
      return std::move(err);
    }

    if (!options.mainFuncName.has_value()) {
      return StreamStringError("Need to have a funcname to compile library");
    }
    this->funcName = options.mainFuncName.value();

    auto result = std::make_unique<LibraryCompilationResult>();
    result->outputDirPath = outputPath;
    result->funcName = *options.mainFuncName;
    return std::move(result);
  }

  /// Load the server lambda from the compilation result.
  llvm::Expected<::concretelang::serverlib::ServerLambda>
  loadServerLambda(LibraryCompilationResult &result, bool useSimulation) {
    EXPECTED_TRY(auto programInfo, getProgramInfo());
    EXPECTED_TRY(ServerProgram serverProgram,
                 outcomeToExpected(ServerProgram::load(programInfo.asReader(),
                                                       getSharedLibPath(),
                                                       useSimulation)));
    EXPECTED_TRY(
        ServerCircuit serverCircuit,
        outcomeToExpected(serverProgram.getServerCircuit(result.funcName)));
    return ::concretelang::serverlib::ServerLambda{serverCircuit,
                                                   useSimulation};
  }

  /// Load the client parameters from the compilation result.
  llvm::Expected<::concretelang::clientlib::ClientParameters>
  loadClientParameters(LibraryCompilationResult &result) {
    EXPECTED_TRY(auto programInfo, getProgramInfo());
    if (programInfo.asReader().getCircuits().size() > 1) {
      return StreamStringError("ClientLambda: Provided program info contains "
                               "more than one circuit.");
    }
    if (programInfo.asReader().getCircuits()[0].getName() != result.funcName) {
      return StreamStringError("Unexpected circuit name in program info");
    }
    auto secretKeys =
        std::vector<::concretelang::clientlib::LweSecretKeyParam>();
    for (auto key : programInfo.asReader().getKeyset().getLweSecretKeys()) {
      secretKeys.push_back(::concretelang::clientlib::LweSecretKeyParam{key});
    }
    auto boostrapKeys =
        std::vector<::concretelang::clientlib::BootstrapKeyParam>();
    for (auto key : programInfo.asReader().getKeyset().getLweBootstrapKeys()) {
      boostrapKeys.push_back(::concretelang::clientlib::BootstrapKeyParam{key});
    }
    auto keyswitchKeys =
        std::vector<::concretelang::clientlib::KeyswitchKeyParam>();
    for (auto key : programInfo.asReader().getKeyset().getLweKeyswitchKeys()) {
      keyswitchKeys.push_back(
          ::concretelang::clientlib::KeyswitchKeyParam{key});
    }
    auto packingKeyswitchKeys =
        std::vector<::concretelang::clientlib::PackingKeyswitchKeyParam>();
    for (auto key :
         programInfo.asReader().getKeyset().getPackingKeyswitchKeys()) {
      packingKeyswitchKeys.push_back(
          ::concretelang::clientlib::PackingKeyswitchKeyParam{key});
    }
    return ::concretelang::clientlib::ClientParameters{
        programInfo, secretKeys, boostrapKeys, keyswitchKeys,
        packingKeyswitchKeys};
  }

  llvm::Expected<Message<concreteprotocol::ProgramInfo>> getProgramInfo() {
    auto path = CompilerEngine::Library::getProgramInfoPath(outputPath);
    std::ifstream file(path);
    std::string content((std::istreambuf_iterator<char>(file)),
                        (std::istreambuf_iterator<char>()));
    if (file.fail()) {
      return StreamStringError("Cannot read file: ") << path;
    }
    auto output = Message<concreteprotocol::ProgramInfo>();
    if (output.readJsonFromString(content).has_failure()) {
      return StreamStringError("Cannot read json string.");
    }
    return output;
  }

  /// Load the the compilation result if circuit already compiled
  llvm::Expected<std::unique_ptr<LibraryCompilationResult>>
  loadCompilationResult() {
    auto result = std::make_unique<LibraryCompilationResult>();
    result->outputDirPath = outputPath;
    result->funcName = funcName;
    return std::move(result);
  }

  llvm::Expected<CompilationFeedback>
  loadCompilationFeedback(LibraryCompilationResult &result) {
    auto path = CompilerEngine::Library::getCompilationFeedbackPath(
        result.outputDirPath);
    auto feedback = CompilationFeedback::load(path);
    if (feedback.has_error()) {
      return StreamStringError(feedback.error().mesg);
    }
    return feedback.value();
  }

  /// Call the lambda with the public arguments.
  llvm::Expected<std::unique_ptr<::concretelang::clientlib::PublicResult>>
  serverCall(::concretelang::serverlib::ServerLambda lambda,
             ::concretelang::clientlib::PublicArguments &args,
             ::concretelang::clientlib::EvaluationKeys &evaluationKeys) {
    if (lambda.isSimulation) {
      return mlir::concretelang::StreamStringError(
          "Tried to perform server call on simulation lambda.");
    }
    EXPECTED_TRY(auto output, outcomeToExpected(lambda.circuit.call(
                                  evaluationKeys.keyset, args.values)));
    ::concretelang::clientlib::PublicResult res{output};
    return std::make_unique<::concretelang::clientlib::PublicResult>(
        std::move(res));
  }

  /// Call the lambda with the public arguments.
  llvm::Expected<std::unique_ptr<::concretelang::clientlib::PublicResult>>
  simulate(::concretelang::serverlib::ServerLambda lambda,
           ::concretelang::clientlib::PublicArguments &args) {
    if (!lambda.isSimulation) {
      return mlir::concretelang::StreamStringError(
          "Tried to perform simulation on execution lambda.");
    }
    EXPECTED_TRY(auto output,
                 outcomeToExpected(lambda.circuit.simulate(args.values)));
    ::concretelang::clientlib::PublicResult res{output};
    return std::make_unique<::concretelang::clientlib::PublicResult>(
        std::move(res));
  }

  /// Get path to shared library
  std::string getSharedLibPath() {
    return CompilerEngine::Library::getSharedLibraryPath(outputPath);
  }

  /// Get path to client parameters file
  std::string getProgramInfoPath() {
    return CompilerEngine::Library::getProgramInfoPath(outputPath);
  }

private:
  std::string outputPath;
  std::string funcName;
  std::string runtimeLibraryPath;
  /// Flags to select generated artifacts
  bool generateSharedLib;
  bool generateStaticLib;
  bool generateClientParameters;
  bool generateCompilationFeedback;
};

} // namespace concretelang
} // namespace mlir

#endif
