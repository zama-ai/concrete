// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <fstream>

#include "boost/outcome.h"
#include "llvm/ADT/Hashing.h"

#include "concretelang/ClientLib/ClientParameters.h"

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

std::size_t ClientParameters::hash() {
  std::size_t currentHash = 1;
  for (auto secretKeyParam : secretKeys) {
    hash_(currentHash, secretKeyParam.first);
    secretKeyParam.second.hash(currentHash);
  }
  for (auto bootstrapKeyParam : bootstrapKeys) {
    hash_(currentHash, bootstrapKeyParam.first);
    bootstrapKeyParam.second.hash(currentHash);
  }
  for (auto keyswitchParam : keyswitchKeys) {
    hash_(currentHash, keyswitchParam.first);
    keyswitchParam.second.hash(currentHash);
  }
  return currentHash;
}

llvm::json::Value toJSON(const LweSecretKeyParam &v) {
  llvm::json::Object object{
      {"dimension", v.dimension},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j, LweSecretKeyParam &v,
              llvm::json::Path p) {
  auto obj = j.getAsObject();
  if (obj == nullptr) {
    p.report("should be an object");
    return false;
  }
  auto dimension = obj->getInteger("dimension");
  if (!dimension.hasValue()) {
    p.report("missing size field");
    return false;
  }
  v.dimension = *dimension;
  return true;
}

llvm::json::Value toJSON(const BootstrapKeyParam &v) {
  llvm::json::Object object{
      {"inputSecretKeyID", v.inputSecretKeyID},
      {"outputSecretKeyID", v.outputSecretKeyID},
      {"level", v.level},
      {"glweDimension", v.glweDimension},
      {"baseLog", v.baseLog},
      {"variance", v.variance},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j, BootstrapKeyParam &v,
              llvm::json::Path p) {
  auto obj = j.getAsObject();
  if (obj == nullptr) {
    p.report("should be an object");
    return false;
  }
  auto inputSecretKeyID = obj->getString("inputSecretKeyID");
  if (!inputSecretKeyID.hasValue()) {
    p.report("missing inputSecretKeyID field");
    return false;
  }
  auto outputSecretKeyID = obj->getString("outputSecretKeyID");
  if (!outputSecretKeyID.hasValue()) {
    p.report("missing outputSecretKeyID field");
    return false;
  }
  auto level = obj->getInteger("level");
  if (!level.hasValue()) {
    p.report("missing level field");
    return false;
  }
  auto baseLog = obj->getInteger("baseLog");
  if (!baseLog.hasValue()) {
    p.report("missing baseLog field");
    return false;
  }
  auto glweDimension = obj->getInteger("glweDimension");
  if (!glweDimension.hasValue()) {
    p.report("missing glweDimension field");
    return false;
  }
  auto variance = obj->getNumber("variance");
  if (!variance.hasValue()) {
    p.report("missing variance field");
    return false;
  }
  v.inputSecretKeyID = (std::string)inputSecretKeyID.getValue();
  v.outputSecretKeyID = (std::string)outputSecretKeyID.getValue();
  v.level = level.getValue();
  v.baseLog = baseLog.getValue();
  v.glweDimension = glweDimension.getValue();
  v.variance = variance.getValue();
  return true;
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
  auto obj = j.getAsObject();
  if (obj == nullptr) {
    p.report("should be an object");
    return false;
  }
  auto inputSecretKeyID = obj->getString("inputSecretKeyID");
  if (!inputSecretKeyID.hasValue()) {
    p.report("missing inputSecretKeyID field");
    return false;
  }
  auto outputSecretKeyID = obj->getString("outputSecretKeyID");
  if (!outputSecretKeyID.hasValue()) {
    p.report("missing outputSecretKeyID field");
    return false;
  }
  auto level = obj->getInteger("level");
  if (!level.hasValue()) {
    p.report("missing level field");
    return false;
  }
  auto baseLog = obj->getInteger("baseLog");
  if (!baseLog.hasValue()) {
    p.report("missing baseLog field");
    return false;
  }
  auto variance = obj->getNumber("variance");
  if (!variance.hasValue()) {
    p.report("missing variance field");
    return false;
  }
  v.inputSecretKeyID = (std::string)inputSecretKeyID.getValue();
  v.outputSecretKeyID = (std::string)outputSecretKeyID.getValue();
  v.level = level.getValue();
  v.baseLog = baseLog.getValue();
  v.variance = variance.getValue();
  return true;
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
  auto obj = j.getAsObject();
  if (obj == nullptr) {
    p.report("should be an object");
    return false;
  }
  auto width = obj->getInteger("width");
  if (!width.hasValue()) {
    p.report("missing width field");
    return false;
  }
  auto dimensions = obj->getArray("dimensions");
  if (dimensions == nullptr) {
    p.report("missing dimensions field");
    return false;
  }
  for (auto dim : *dimensions) {
    auto iDim = dim.getAsInteger();
    if (!iDim.hasValue()) {
      p.report("dimensions must be integer");
      return false;
    }
    v.dimensions.push_back(iDim.getValue());
  }
  auto size = obj->getInteger("size");
  if (!size.hasValue()) {
    p.report("missing size field");
    return false;
  }
  auto sign = obj->getBoolean("sign");
  if (!sign.hasValue()) {
    p.report("missing sign field");
    return false;
  }
  v.width = width.getValue();
  v.size = size.getValue();
  v.sign = sign.getValue();
  return true;
}

llvm::json::Value toJSON(const Encoding &v) {
  llvm::json::Object object{
      {"precision", v.precision},
  };
  if (!v.crt.empty()) {
    object.insert({"crt", v.crt});
  }
  return object;
}
bool fromJSON(const llvm::json::Value j, Encoding &v, llvm::json::Path p) {
  auto obj = j.getAsObject();
  if (obj == nullptr) {
    p.report("should be an object");
    return false;
  }
  auto precision = obj->getInteger("precision");
  if (!precision.hasValue()) {
    p.report("missing precision field");
    return false;
  }
  v.precision = precision.getValue();
  auto crt = obj->getArray("crt");
  if (crt != nullptr) {
    for (auto dim : *crt) {
      auto iDim = dim.getAsInteger();
      if (!iDim.hasValue()) {
        p.report("dimensions must be integer");
        return false;
      }
      v.crt.push_back(iDim.getValue());
    }
  }

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
  auto obj = j.getAsObject();
  if (obj == nullptr) {
    p.report("should be an object");
    return false;
  }
  auto secretKeyID = obj->getString("secretKeyID");
  if (!secretKeyID.hasValue()) {
    p.report("missing secretKeyID field");
    return false;
  }
  v.secretKeyID = (std::string)secretKeyID.getValue();
  auto variance = obj->getNumber("variance");
  if (!variance.hasValue()) {
    p.report("missing variance field");
    return false;
  }
  v.variance = variance.getValue();
  auto encoding = obj->get("encoding");
  if (encoding == nullptr) {
    p.report("missing encoding field");
    return false;
  }
  if (!fromJSON(*encoding, v.encoding, p.field("encoding"))) {
    return false;
  }
  return true;
}

llvm::json::Value toJSON(const CircuitGate &v) {
  llvm::json::Object object{
      {"encryption", v.encryption},
      {"shape", v.shape},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, CircuitGate &v, llvm::json::Path p) {
  auto obj = j.getAsObject();
  auto encryption = obj->get("encryption");
  if (encryption == nullptr) {
    p.report("missing encryption field");
    return false;
  }
  if (!fromJSON(*encryption, v.encryption, p.field("encryption"))) {
    return false;
  }
  auto shape = obj->get("shape");
  if (shape == nullptr) {
    p.report("missing shape field");
    return false;
  }
  if (!fromJSON(*shape, v.shape, p.field("shape"))) {
    return false;
  }
  return true;
}

template <typename T> llvm::json::Value toJson(std::map<std::string, T> map) {
  llvm::json::Object obj;
  for (auto entry : map) {
    obj[entry.first] = entry.second;
  }
  return obj;
}

llvm::json::Value toJSON(const ClientParameters &v) {
  llvm::json::Object object{
      {"secretKeys", toJson(v.secretKeys)},
      {"bootstrapKeys", toJson(v.bootstrapKeys)},
      {"keyswitchKeys", toJson(v.keyswitchKeys)},
      {"inputs", v.inputs},
      {"outputs", v.outputs},
      {"functionName", v.functionName},
  };
  return object;
}
bool fromJSON(const llvm::json::Value j, ClientParameters &v,
              llvm::json::Path p) {

  auto obj = j.getAsObject();
  auto secretkeys = obj->get("secretKeys");
  if (secretkeys == nullptr) {
    p.report("missing secretKeys field");
    return false;
  }
  if (!fromJSON(*secretkeys, v.secretKeys, p.field("secretKeys"))) {
    return false;
  }
  auto bootstrapKeys = obj->get("bootstrapKeys");
  if (bootstrapKeys == nullptr) {
    p.report("missing bootstrapKeys field");
    return false;
  }
  if (!fromJSON(*bootstrapKeys, v.bootstrapKeys, p.field("bootstrapKeys"))) {
    return false;
  }
  auto keyswitchKeys = obj->get("keyswitchKeys");
  if (keyswitchKeys == nullptr) {
    p.report("missing keyswitchKeys field");
    return false;
  }
  if (!fromJSON(*keyswitchKeys, v.keyswitchKeys, p.field("keyswitchKeys"))) {
    return false;
  }
  auto inputs = obj->get("inputs");
  if (inputs == nullptr) {
    p.report("missing inputs field");
    return false;
  }
  if (!fromJSON(*inputs, v.inputs, p.field("inputs"))) {
    return false;
  }
  auto outputs = obj->get("outputs");
  if (outputs == nullptr) {
    p.report("missing outputs field");
    return false;
  }
  if (!fromJSON(*outputs, v.outputs, p.field("outputs"))) {
    return false;
  }
  auto functionName = obj->getString("functionName");
  if (!functionName.hasValue()) {
    p.report("missing functionName field");
    return false;
  }
  v.functionName = (std::string)functionName.getValue();

  return true;
}

std::string ClientParameters::getClientParametersPath(std::string path) {
  return path + CLIENT_PARAMETERS_EXT;
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
