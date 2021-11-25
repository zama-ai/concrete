#ifndef ZAMALANG_SUPPORT_CLIENTPARAMETERS_H_
#define ZAMALANG_SUPPORT_CLIENTPARAMETERS_H_
#include <map>
#include <string>
#include <vector>

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>

#include "zamalang/Support/V0Parameters.h"

namespace mlir {
namespace zamalang {

typedef size_t DecompositionLevelCount;
typedef size_t DecompositionBaseLog;
typedef size_t PolynomialSize;
typedef size_t Precision;
typedef double Variance;

typedef uint64_t LweSize;
typedef uint64_t GlweDimension;

typedef std::string LweSecretKeyID;
struct LweSecretKeyParam {
  LweSize size;

  void hash(size_t &seed);
};

typedef std::string BootstrapKeyID;
struct BootstrapKeyParam {
  LweSecretKeyID inputSecretKeyID;
  LweSecretKeyID outputSecretKeyID;
  DecompositionLevelCount level;
  DecompositionBaseLog baseLog;
  GlweDimension glweDimension;
  Variance variance;

  void hash(size_t &seed);
};

typedef std::string KeyswitchKeyID;
struct KeyswitchKeyParam {
  LweSecretKeyID inputSecretKeyID;
  LweSecretKeyID outputSecretKeyID;
  DecompositionLevelCount level;
  DecompositionBaseLog baseLog;
  Variance variance;

  void hash(size_t &seed);
};

struct Encoding {
  Precision precision;
};

struct EncryptionGate {
  LweSecretKeyID secretKeyID;
  Variance variance;
  Encoding encoding;
};

struct CircuitGateShape {
  // Width of the scalar value
  size_t width;
  // Dimensions of the tensor, empty if scalar
  std::vector<int64_t> dimensions;
  // Size of the buffer containing the tensor
  size_t size;
};

struct CircuitGate {
  llvm::Optional<EncryptionGate> encryption;
  CircuitGateShape shape;
};

struct ClientParameters {
  std::map<LweSecretKeyID, LweSecretKeyParam> secretKeys;
  std::map<BootstrapKeyID, BootstrapKeyParam> bootstrapKeys;
  std::map<KeyswitchKeyID, KeyswitchKeyParam> keyswitchKeys;
  std::vector<CircuitGate> inputs;
  std::vector<CircuitGate> outputs;
  size_t hash();
};

llvm::Expected<ClientParameters>
createClientParametersForV0(V0FHEContext context, llvm::StringRef name,
                            mlir::ModuleOp module);

} // namespace zamalang
} // namespace mlir

#endif