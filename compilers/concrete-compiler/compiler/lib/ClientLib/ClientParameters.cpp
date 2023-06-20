// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <fstream>
#include <optional>
#include <vector>

#include "boost/outcome.h"
#include "llvm/ADT/Hashing.h"

#include "concrete-protocol.pb.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Common/Protobuf.h"

namespace concretelang {
namespace clientlib {

using StringError = concretelang::error::StringError;

// https://stackoverflow.com/a/38140932
static inline void hash_(std::size_t &seed) {}
template <typename T, typename... Rest>
static inline void hash_(std::size_t &seed, const T &v, Rest... rest) {
  // See https://softwareengineering.stackexchange.com/a/402543
  const auto GOLDEN_RATIO = 0x9e3779b97f4a7c15; // pseudo random bits
  seed ^= llvm::hash_value(v) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
  hash_(seed, rest...);
}

static long double_to_bits(double &v) { return *reinterpret_cast<long *>(&v); }

void LweSecretKeyParam::hash(size_t &seed) { hash_(seed, dimension); }

void BootstrapKeyParam::hash(size_t &seed) {
  hash_(seed, inputSecretKeyID, outputSecretKeyID, level, baseLog,
        glweDimension, double_to_bits(variance));
}

void KeyswitchKeyParam::hash(size_t &seed) {
  hash_(seed, inputSecretKeyID, outputSecretKeyID, level, baseLog,
        double_to_bits(variance));
}

void PackingKeyswitchKeyParam::hash(size_t &seed) {
  hash_(seed, inputSecretKeyID, outputSecretKeyID, level, baseLog,
        glweDimension, polynomialSize, inputLweDimension,
        double_to_bits(variance));
}

std::size_t ClientParameters::hash() {
  std::size_t currentHash = 1;
  for (auto secretKeyParam : secretKeys) {
    secretKeyParam.hash(currentHash);
  }
  for (auto bootstrapKeyParam : bootstrapKeys) {
    bootstrapKeyParam.hash(currentHash);
  }
  for (auto keyswitchParam : keyswitchKeys) {
    keyswitchParam.hash(currentHash);
  }
  for (auto packingKeyswitchKeyParam : packingKeyswitchKeys) {
    packingKeyswitchKeyParam.hash(currentHash);
  }
  return currentHash;
}

std::string ClientParameters::getClientParametersPath(std::string path) {
  return path + CLIENT_PARAMETERS_EXT;
}

typedef std::tuple<
    std::vector<LweSecretKeyParam>, std::vector<BootstrapKeyParam>,
    std::vector<KeyswitchKeyParam>, std::vector<PackingKeyswitchKeyParam>>
    ClientParametersKeysetParams;

ClientParametersKeysetParams
extractKeysetParams(const concreteprotocol::KeysetInfo &keysetInfo) {
  std::vector<LweSecretKeyParam> secretKeys;
  for (auto keyInfo : keysetInfo.lwesecretkeys()) {
    secretKeys.push_back(LweSecretKeyParam{keyInfo.params().lwedimension()});
  }

  std::vector<BootstrapKeyParam> bootstrapKeys;
  for (auto keyInfo : keysetInfo.lwebootstrapkeys()) {
    bootstrapKeys.push_back(BootstrapKeyParam{
        keyInfo.inputid(), keyInfo.outputid(), keyInfo.params().levelcount(),
        keyInfo.params().baselog(), keyInfo.params().glwedimension(),
        keyInfo.params().variance(), keyInfo.params().polynomialsize(),
        secretKeys[keyInfo.inputid()].lweDimension()});
  }

  std::vector<KeyswitchKeyParam> keyswitchKeys;
  for (auto keyInfo : keysetInfo.lwekeyswitchkeys()) {
    keyswitchKeys.push_back(KeyswitchKeyParam{
        keyInfo.inputid(),
        keyInfo.outputid(),
        keyInfo.params().levelcount(),
        keyInfo.params().baselog(),
        keyInfo.params().variance(),
    });
  }

  std::vector<PackingKeyswitchKeyParam> packingKeyswitchKeys;
  for (auto keyInfo : keysetInfo.packingkeyswitchkeys()) {
    packingKeyswitchKeys.push_back(PackingKeyswitchKeyParam{
        keyInfo.inputid(), keyInfo.outputid(), keyInfo.params().levelcount(),
        keyInfo.params().baselog(), keyInfo.params().glwedimension(),
        keyInfo.params().polynomialsize(),
        secretKeys[keyInfo.inputid()].lweDimension(),
        keyInfo.params().variance()});
  }

  return {secretKeys, bootstrapKeys, keyswitchKeys, packingKeyswitchKeys};
}

CircuitGate gateInfoToCircuitGate(concreteprotocol::GateInfo &gateInfo) {

  uint64_t size = 0;
  if (gateInfo.shape().dimensions_size() > 0) {
    size = 1;
    for (auto dim : gateInfo.shape().dimensions()) {
      size *= dim;
    }
  }
  std::vector<int64_t> dims{gateInfo.shape().dimensions().begin(),
                            gateInfo.shape().dimensions().end()};

  if (gateInfo.has_lweciphertext() && gateInfo.lweciphertext().has_integer()) {
    auto encryptionInfo = gateInfo.lweciphertext().encryption();
    auto encodingInfo = gateInfo.lweciphertext().integer();
    size_t width = encodingInfo.width();
    bool isSigned = encodingInfo.issigned();
    LweSecretKeyID secretKeyID = encryptionInfo.keyid();
    Variance variance = encryptionInfo.variance();
    CRTDecomposition crt = std::vector<int64_t>();
    std::optional<ChunkInfo> chunkInfo = std::nullopt;
    if (encodingInfo.has_chunked()) {
      auto chunkedMode = encodingInfo.chunked();
      width = chunkedMode.width();
      chunkInfo = ChunkInfo{chunkedMode.size(), chunkedMode.width()};
    }
    if (encodingInfo.has_crt()) {
      auto crtMode = encodingInfo.crt();
      crt = std::vector<int64_t>(crtMode.moduli().begin(),
                                 crtMode.moduli().end());
    }
    return CircuitGate{
        /* .encryption = */ std::optional<EncryptionGate>({
            /* .secretKeyID = */ secretKeyID,
            /* .variance = */ variance,
            /* .encoding = */
            {
                /* .precision = */ width,
                /* .crt = */ crt,
                /*.sign = */ isSigned,
            },
        }),
        /*.shape = */
        {
            /*.width = */ width,
            /*.dimensions = */ dims,
            /*.size = */ size,
            /*.sign = */ isSigned,
        },
        /*.chunkInfo = */ chunkInfo,
    };
  } else if (gateInfo.has_lweciphertext() &&
             gateInfo.lweciphertext().has_boolean()) {
    auto encryptionInfo = gateInfo.lweciphertext().encryption();
    auto encodingInfo = gateInfo.lweciphertext().boolean();
    size_t width = 2;
    bool isSigned = false;
    LweSecretKeyID secretKeyID = encryptionInfo.keyid();
    Variance variance = encryptionInfo.variance();
    CRTDecomposition crt = std::vector<int64_t>();
    return CircuitGate{
        /* .encryption = */ std::optional<EncryptionGate>({
            /* .secretKeyID = */ secretKeyID,
            /* .variance = */ variance,
            /* .encoding = */
            {
                /* .precision = */ width,
                /* .crt = */ crt,
                /*.sign = */ isSigned,
            },
        }),
        /*.shape = */
        {
            /*.width = */ width,
            /*.dimensions = */ dims,
            /*.size = */ size,
            /*.sign = */ isSigned,
        },
        /*.chunkInfo = */ std::nullopt,
    };
  } else if (gateInfo.has_plaintext()) {
    auto encodingInfo = gateInfo.plaintext();
    size_t width = encodingInfo.integerprecision();
    bool sign = encodingInfo.issigned();
    return CircuitGate{
        /*.encryption = */ std::nullopt,
        /*.shape = */
        {/*.width = */ width,
         /*.dimensions = */ dims,
         /*.size = */ size,
         /* .sign */ sign},
        /*.chunkInfo = */ std::nullopt,
    };
  } else if (gateInfo.has_index()) {
    auto encodingInfo = gateInfo.index();
    size_t width = encodingInfo.integerprecision();
    bool sign = encodingInfo.issigned();
    return CircuitGate{
        /*.encryption = */ std::nullopt,
        /*.shape = */
        {/*.width = */ width,
         /*.dimensions = */ dims,
         /*.size = */ size,
         /* .sign */ sign},
        /*.chunkInfo = */ std::nullopt,
    };
  } else {
    assert(false && "Fatal error");
  }
}

typedef std::tuple<std::vector<CircuitGate>, std::vector<CircuitGate>,
                   std::string>
    ClientParametersCircuitParams;

ClientParametersCircuitParams
extractCircuitParams(const concreteprotocol::CircuitInfo &circuitInfo) {
  std::vector<CircuitGate> inputs;
  for (auto gateInfo : circuitInfo.inputs()) {
    inputs.push_back(gateInfoToCircuitGate(gateInfo));
  }
  std::vector<CircuitGate> outputs;
  for (auto gateInfo : circuitInfo.outputs()) {
    outputs.push_back(gateInfoToCircuitGate(gateInfo));
  }
  return {inputs, outputs, circuitInfo.name()};
}

ClientParameters ClientParameters::fromProgramInfo(
    const concreteprotocol::ProgramInfo &programInfo) {
  auto output = ClientParameters{};
  auto keysetParams = extractKeysetParams(programInfo.keyset());
  auto circuitParams = extractCircuitParams(programInfo.circuits(0));
  output.secretKeys = std::get<0>(keysetParams);
  output.bootstrapKeys = std::get<1>(keysetParams);
  output.keyswitchKeys = std::get<2>(keysetParams);
  output.packingKeyswitchKeys = std::get<3>(keysetParams);
  output.inputs = std::get<0>(circuitParams);
  output.outputs = std::get<1>(circuitParams);
  output.functionName = std::get<2>(circuitParams);
  return output;
}

llvm::json::Value toJSON(const LweSecretKeyParam &v) {
  llvm::json::Object object{
      {"dimension", v.dimension},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j, LweSecretKeyParam &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("dimension", v.dimension);
}

llvm::json::Value toJSON(const BootstrapKeyParam &v) {
  llvm::json::Object object{
      {"inputSecretKeyID", v.inputSecretKeyID},
      {"outputSecretKeyID", v.outputSecretKeyID},
      {"level", v.level},
      {"glweDimension", v.glweDimension},
      {"baseLog", v.baseLog},
      {"variance", v.variance},
      {"polynomialSize", v.polynomialSize},
      {"inputLweDimension", v.inputLweDimension},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j, BootstrapKeyParam &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("inputSecretKeyID", v.inputSecretKeyID) &&
         O.map("outputSecretKeyID", v.outputSecretKeyID) &&
         O.map("level", v.level) && O.map("baseLog", v.baseLog) &&
         O.map("glweDimension", v.glweDimension) &&
         O.map("variance", v.variance) &&
         O.map("polynomialSize", v.polynomialSize) &&
         O.map("inputLweDimension", v.inputLweDimension);
}

llvm::json::Value toJSON(const KeyswitchKeyParam &v) {
  llvm::json::Object object{
      {"inputSecretKeyID", v.inputSecretKeyID},
      {"outputSecretKeyID", v.outputSecretKeyID},
      {"level", v.level},
      {"baseLog", v.baseLog},
      {"variance", v.variance},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, KeyswitchKeyParam &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("inputSecretKeyID", v.inputSecretKeyID) &&
         O.map("outputSecretKeyID", v.outputSecretKeyID) &&
         O.map("level", v.level) && O.map("baseLog", v.baseLog) &&
         O.map("variance", v.variance);
}

llvm::json::Value toJSON(const PackingKeyswitchKeyParam &v) {
  llvm::json::Object object{
      {"inputSecretKeyID", v.inputSecretKeyID},
      {"outputSecretKeyID", v.outputSecretKeyID},
      {"level", v.level},
      {"baseLog", v.baseLog},
      {"glweDimension", v.glweDimension},
      {"polynomialSize", v.polynomialSize},
      {"inputLweDimension", v.inputLweDimension},
      {"variance", v.variance},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, PackingKeyswitchKeyParam &v,
              llvm::json::Path p) {

  llvm::json::ObjectMapper O(j, p);
  return O && O.map("inputSecretKeyID", v.inputSecretKeyID) &&
         O.map("outputSecretKeyID", v.outputSecretKeyID) &&
         O.map("level", v.level) && O.map("baseLog", v.baseLog) &&
         O.map("glweDimension", v.glweDimension) &&
         O.map("polynomialSize", v.polynomialSize) &&
         O.map("inputLweDimension", v.inputLweDimension) &&
         O.map("variance", v.variance);
}

llvm::json::Value toJSON(const CircuitGateShape &v) {
  llvm::json::Object object{
      {"width", v.width},
      {"dimensions", v.dimensions},
      {"size", v.size},
      {"sign", v.sign},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, CircuitGateShape &v,
              llvm::json::Path p) {

  llvm::json::ObjectMapper O(j, p);
  return O && O.map("width", v.width) && O.map("size", v.size) &&
         O.map("dimensions", v.dimensions) && O.map("sign", v.sign);
}

llvm::json::Value toJSON(const Encoding &v) {
  llvm::json::Object object{
      {"precision", v.precision},
      {"isSigned", v.isSigned},
  };
  if (!v.crt.empty()) {
    object.insert({"crt", v.crt});
  }
  return object;
}
bool fromJSON(const llvm::json::Value j, Encoding &v, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  if (!(O && O.map("precision", v.precision) &&
        O.map("isSigned", v.isSigned))) {
    return false;
  }
  // TODO: check this is correct for an optional field
  O.map("crt", v.crt);
  return true;
}

llvm::json::Value toJSON(const EncryptionGate &v) {
  llvm::json::Object object{
      {"secretKeyID", v.secretKeyID},
      {"variance", v.variance},
      {"encoding", v.encoding},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, EncryptionGate &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("secretKeyID", v.secretKeyID) &&
         O.map("variance", v.variance) && O.map("encoding", v.encoding);
}

llvm::json::Value toJSON(const CircuitGate &v) {
  llvm::json::Object object{
      {"encryption", v.encryption},
      {"shape", v.shape},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, CircuitGate &v, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("encryption", v.encryption) && O.map("shape", v.shape);
}

llvm::json::Value toJSON(const ClientParameters &v) {
  llvm::json::Object object{
      {"secretKeys", v.secretKeys},
      {"bootstrapKeys", v.bootstrapKeys},
      {"keyswitchKeys", v.keyswitchKeys},
      {"packingKeyswitchKeys", v.packingKeyswitchKeys},
      {"inputs", v.inputs},
      {"outputs", v.outputs},
      {"functionName", v.functionName},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, ClientParameters &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("secretKeys", v.secretKeys) &&
         O.map("bootstrapKeys", v.bootstrapKeys) &&
         O.map("keyswitchKeys", v.keyswitchKeys) &&
         O.map("packingKeyswitchKeys", v.packingKeyswitchKeys) &&
         O.map("inputs", v.inputs) && O.map("outputs", v.outputs) &&
         O.map("functionName", v.functionName);
}

outcome::checked<std::vector<ClientParameters>, StringError>
ClientParameters::load(std::string jsonPath) {
  std::ifstream file(jsonPath);
  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));
  if (file.fail()) {
    return StringError("Cannot read file: ") << jsonPath;
  }
  auto expectedClientParams =
      llvm::json::parse<std::vector<ClientParameters>>(content);
  if (auto err = expectedClientParams.takeError()) {
    return StringError("Cannot open client parameters: ")
           << llvm::toString(std::move(err)) << "\n"
           << content << "\n";
  }
  return expectedClientParams.get();
}

} // namespace clientlib
} // namespace concretelang
