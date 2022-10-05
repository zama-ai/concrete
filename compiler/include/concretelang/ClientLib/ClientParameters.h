// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_CLIENTPARAMETERS_H_
#define CONCRETELANG_CLIENTLIB_CLIENTPARAMETERS_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "boost/outcome.h"

#include "concretelang/Common/Error.h"

#include <llvm/Support/JSON.h>

namespace concretelang {

inline size_t bitWidthAsWord(size_t exactBitWidth) {
  if (exactBitWidth <= 8)
    return 8;
  if (exactBitWidth <= 16)
    return 16;
  if (exactBitWidth <= 32)
    return 32;
  if (exactBitWidth <= 64)
    return 64;
  assert(false && "Bit witdh > 64 not supported");
}

namespace clientlib {

using concretelang::error::StringError;

const std::string SMALL_KEY = "small";
const std::string BIG_KEY = "big";
const std::string BOOTSTRAP_KEY = "bsk_v0";
const std::string KEYSWITCH_KEY = "ksk_v0";

const std::string CLIENT_PARAMETERS_EXT = ".concrete.params.json";

typedef size_t DecompositionLevelCount;
typedef size_t DecompositionBaseLog;
typedef size_t PolynomialSize;
typedef size_t Precision;
typedef double Variance;
typedef std::vector<int64_t> CRTDecomposition;

typedef uint64_t LweDimension;
typedef uint64_t GlweDimension;

typedef std::string LweSecretKeyID;
struct LweSecretKeyParam {
  LweDimension dimension;

  void hash(size_t &seed);
  inline uint64_t lweDimension() { return dimension; }
  inline uint64_t lweSize() { return dimension + 1; }
  inline uint64_t byteSize() { return lweSize() * 8; }
};
static bool operator==(const LweSecretKeyParam &lhs,
                       const LweSecretKeyParam &rhs) {
  return lhs.dimension == rhs.dimension;
}

typedef std::string BootstrapKeyID;
struct BootstrapKeyParam {
  LweSecretKeyID inputSecretKeyID;
  LweSecretKeyID outputSecretKeyID;
  DecompositionLevelCount level;
  DecompositionBaseLog baseLog;
  GlweDimension glweDimension;
  Variance variance;

  void hash(size_t &seed);

  uint64_t byteSize(uint64_t inputLweSize, uint64_t outputLweSize) {
    return inputLweSize * level * (glweDimension + 1) * (glweDimension + 1) *
           outputLweSize * 8;
  }
};
static inline bool operator==(const BootstrapKeyParam &lhs,
                              const BootstrapKeyParam &rhs) {
  return lhs.inputSecretKeyID == rhs.inputSecretKeyID &&
         lhs.outputSecretKeyID == rhs.outputSecretKeyID &&
         lhs.level == rhs.level && lhs.baseLog == rhs.baseLog &&
         lhs.glweDimension == rhs.glweDimension && lhs.variance == rhs.variance;
}

typedef std::string KeyswitchKeyID;
struct KeyswitchKeyParam {
  LweSecretKeyID inputSecretKeyID;
  LweSecretKeyID outputSecretKeyID;
  DecompositionLevelCount level;
  DecompositionBaseLog baseLog;
  Variance variance;

  void hash(size_t &seed);

  size_t byteSize(size_t inputLweSize, size_t outputLweSize) {
    return level * inputLweSize * outputLweSize * 8;
  }
};
static inline bool operator==(const KeyswitchKeyParam &lhs,
                              const KeyswitchKeyParam &rhs) {
  return lhs.inputSecretKeyID == rhs.inputSecretKeyID &&
         lhs.outputSecretKeyID == rhs.outputSecretKeyID &&
         lhs.level == rhs.level && lhs.baseLog == rhs.baseLog &&
         lhs.variance == rhs.variance;
}

typedef std::string PackingKeySwitchID;
struct PackingKeySwitchParam {
  LweSecretKeyID inputSecretKeyID;
  LweSecretKeyID outputSecretKeyID;
  BootstrapKeyID bootstrapKeyID;
  size_t level;
  size_t baseLog;
  Variance variance;

  void hash(size_t &seed);
};
static inline bool operator==(const PackingKeySwitchParam &lhs,
                              const PackingKeySwitchParam &rhs) {
  return lhs.inputSecretKeyID == rhs.inputSecretKeyID &&
         lhs.outputSecretKeyID == rhs.outputSecretKeyID &&
         lhs.level == rhs.level && lhs.baseLog == rhs.baseLog;
}

struct Encoding {
  Precision precision;
  CRTDecomposition crt;
};
static inline bool operator==(const Encoding &lhs, const Encoding &rhs) {
  return lhs.precision == rhs.precision;
}

struct EncryptionGate {
  LweSecretKeyID secretKeyID;
  Variance variance;
  Encoding encoding;
};
static inline bool operator==(const EncryptionGate &lhs,
                              const EncryptionGate &rhs) {
  return lhs.secretKeyID == rhs.secretKeyID && lhs.variance == rhs.variance &&
         lhs.encoding == rhs.encoding;
}

struct CircuitGateShape {
  /// Width of the scalar value
  size_t width;
  /// Dimensions of the tensor, empty if scalar
  std::vector<int64_t> dimensions;
  /// Size of the buffer containing the tensor
  size_t size;
  // Indicated whether elements are signed
  bool sign;
};
static inline bool operator==(const CircuitGateShape &lhs,
                              const CircuitGateShape &rhs) {
  return lhs.width == rhs.width && lhs.dimensions == rhs.dimensions &&
         lhs.size == rhs.size;
}

struct CircuitGate {
  llvm::Optional<EncryptionGate> encryption;
  CircuitGateShape shape;

  bool isEncrypted() { return encryption.hasValue(); }

  /// byteSize returns the size in bytes for this gate.
  size_t byteSize(std::map<LweSecretKeyID, LweSecretKeyParam> secretKeys) {
    auto width = shape.width;
    auto numElts = shape.size == 0 ? 1 : shape.size;
    if (isEncrypted()) {
      auto skParam = secretKeys.find(encryption->secretKeyID);
      assert(skParam != secretKeys.end());
      return 8 * skParam->second.lweSize() * numElts;
    }
    width = bitWidthAsWord(width) / 8;
    return width * numElts;
  }
};
static inline bool operator==(const CircuitGate &lhs, const CircuitGate &rhs) {
  return lhs.encryption == rhs.encryption && lhs.shape == rhs.shape;
}

struct ClientParameters {
  std::map<LweSecretKeyID, LweSecretKeyParam> secretKeys;
  std::map<BootstrapKeyID, BootstrapKeyParam> bootstrapKeys;
  std::map<KeyswitchKeyID, KeyswitchKeyParam> keyswitchKeys;
  std::map<PackingKeySwitchID, PackingKeySwitchParam> packingKeys;
  std::vector<CircuitGate> inputs;
  std::vector<CircuitGate> outputs;
  std::string functionName;

  size_t hash();

  static outcome::checked<std::vector<ClientParameters>, StringError>
  load(std::string path);

  static std::string getClientParametersPath(std::string path);

  outcome::checked<CircuitGate, StringError> input(size_t pos) {
    if (pos >= inputs.size()) {
      return StringError("input gate ") << pos << " didn't exists";
    }
    return inputs[pos];
  }

  outcome::checked<CircuitGate, StringError> ouput(size_t pos) {
    if (pos >= outputs.size()) {
      return StringError("output gate ") << pos << " didn't exists";
    }
    return outputs[pos];
  }

  outcome::checked<LweSecretKeyParam, StringError>
  lweSecretKeyParam(CircuitGate gate) {
    if (!gate.encryption.hasValue()) {
      return StringError("gate is not encrypted");
    }
    auto secretKey = secretKeys.find(gate.encryption->secretKeyID);
    if (secretKey == secretKeys.end()) {
      return StringError("cannot find ")
             << gate.encryption->secretKeyID << " in client parameters";
    }
    return secretKey->second;
  }

  /// bufferSize returns the size of the whole buffer of a gate.
  int64_t bufferSize(CircuitGate gate) {
    if (!gate.encryption.hasValue()) {
      // Value is not encrypted just returns the tensor size
      return gate.shape.size;
    }
    auto shapeSize = gate.shape.size == 0 ? 1 : gate.shape.size;
    // Size of the ciphertext
    return shapeSize * lweBufferSize(gate);
  }

  /// lweBufferSize returns the size of one ciphertext of a gate.
  int64_t lweBufferSize(CircuitGate gate) {
    assert(gate.encryption.hasValue());
    auto nbBlocks = gate.encryption->encoding.crt.size();
    nbBlocks = nbBlocks == 0 ? 1 : nbBlocks;

    auto param = lweSecretKeyParam(gate);
    assert(param.has_value());
    return param.value().lweSize() * nbBlocks;
  }

  /// bufferShape returns the shape of the tensor for the given gate. It returns
  /// the shape used at low-level, i.e. contains the dimensions for ciphertexts.
  std::vector<int64_t> bufferShape(CircuitGate gate) {
    if (!gate.encryption.hasValue()) {
      // Value is not encrypted just returns the tensor shape
      return gate.shape.dimensions;
    }
    auto lweSecreteKeyParam = lweSecretKeyParam(gate);
    assert(lweSecreteKeyParam.has_value());

    // Copy the shape
    std::vector<int64_t> shape(gate.shape.dimensions);

    auto crt = gate.encryption->encoding.crt;

    // CRT case: Add one dimension equals to the number of blocks
    if (!crt.empty()) {
      shape.push_back(crt.size());
    }
    // Add one dimension for the size of ciphertext(s)
    shape.push_back(lweSecreteKeyParam.value().lweSize());
    return shape;
  }
};

static inline bool operator==(const ClientParameters &lhs,
                              const ClientParameters &rhs) {
  return lhs.secretKeys == rhs.secretKeys &&
         lhs.bootstrapKeys == rhs.bootstrapKeys &&
         lhs.keyswitchKeys == rhs.keyswitchKeys && lhs.inputs == lhs.inputs &&
         lhs.outputs == lhs.outputs;
}

llvm::json::Value toJSON(const LweSecretKeyParam &);
bool fromJSON(const llvm::json::Value, LweSecretKeyParam &, llvm::json::Path);

llvm::json::Value toJSON(const BootstrapKeyParam &);
bool fromJSON(const llvm::json::Value, BootstrapKeyParam &, llvm::json::Path);

llvm::json::Value toJSON(const KeyswitchKeyParam &);
bool fromJSON(const llvm::json::Value, KeyswitchKeyParam &, llvm::json::Path);

llvm::json::Value toJSON(const PackingKeySwitchParam &);
bool fromJSON(const llvm::json::Value, PackingKeySwitchParam &,
              llvm::json::Path);

llvm::json::Value toJSON(const Encoding &);
bool fromJSON(const llvm::json::Value, Encoding &, llvm::json::Path);

llvm::json::Value toJSON(const EncryptionGate &);
bool fromJSON(const llvm::json::Value, EncryptionGate &, llvm::json::Path);

llvm::json::Value toJSON(const CircuitGateShape &);
bool fromJSON(const llvm::json::Value, CircuitGateShape &, llvm::json::Path);

llvm::json::Value toJSON(const CircuitGate &);
bool fromJSON(const llvm::json::Value, CircuitGate &, llvm::json::Path);

llvm::json::Value toJSON(const ClientParameters &);
bool fromJSON(const llvm::json::Value, ClientParameters &, llvm::json::Path);

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                            ClientParameters cp) {
  return OS << llvm::formatv("{0:2}", toJSON(cp));
}

static inline llvm::raw_ostream &operator<<(llvm::raw_string_ostream &OS,
                                            ClientParameters cp) {
  return OS << llvm::formatv("{0:2}", toJSON(cp));
}

} // namespace clientlib
} // namespace concretelang

#endif
