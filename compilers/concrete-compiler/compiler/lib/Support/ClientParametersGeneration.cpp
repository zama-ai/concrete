// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.
#include <cassert>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <optional>
#include <unordered_set>
#include <variant>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/Error.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "concrete/curves.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Support/Encodings.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/TFHECircuitKeys.h"
#include "concretelang/Support/Variants.h"
#include "llvm/Config/abi-breaking.h"

namespace mlir {
namespace concretelang {

namespace clientlib = ::concretelang::clientlib;
using ::concretelang::clientlib::ChunkInfo;
using ::concretelang::clientlib::CircuitGate;
using ::concretelang::clientlib::ClientParameters;
using ::concretelang::clientlib::Encoding;
using ::concretelang::clientlib::EncryptionGate;
using ::concretelang::clientlib::LweSecretKeyID;
using ::concretelang::clientlib::Precision;
using ::concretelang::clientlib::Variance;

const auto keyFormat = concrete::BINARY;

llvm::Expected<CircuitGate>
generateGate(mlir::Type type, encodings::Encoding encoding,
             concrete::SecurityCurve curve,
             std::optional<CRTDecomposition> maybeCrt) {
  auto scalarVisitor = overloaded{
      [&](encodings::EncryptedIntegerScalarEncoding enc)
          -> llvm::Expected<CircuitGate> {
        TFHE::GLWESecretKeyNormalized normKey;
        if (type.isa<RankedTensorType>()) {
          normKey = type.cast<RankedTensorType>()
                        .getElementType()
                        .cast<TFHE::GLWECipherTextType>()
                        .getKey()
                        .getNormalized()
                        .value();
        } else {
          normKey = type.cast<TFHE::GLWECipherTextType>()
                        .getKey()
                        .getNormalized()
                        .value();
        }
        if ((int)normKey.dimension < curve.minimalLweDimension) {
          return llvm::make_error<llvm::StringError>(
              "Minimal size for security is not attained",
              llvm::inconvertibleErrorCode());
        }
        size_t width = enc.width;
        bool isSigned = enc.isSigned;
        uint64_t size = 0;
        std::vector<int64_t> dims{};
        LweSecretKeyID secretKeyID = normKey.index;
        Variance variance = curve.getVariance(1, normKey.dimension, 64);
        CRTDecomposition crt = maybeCrt.value_or(std::vector<int64_t>());
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
      },
      [&](encodings::EncryptedChunkedIntegerScalarEncoding enc)
          -> llvm::Expected<CircuitGate> {
        auto tensorType = type.cast<mlir::RankedTensorType>();
        auto glweType =
            tensorType.getElementType().cast<TFHE::GLWECipherTextType>();
        auto normKey = glweType.getKey().getNormalized().value();
        if ((int)normKey.dimension < curve.minimalLweDimension) {
          return llvm::make_error<llvm::StringError>(
              "Minimal size for security is not attained",
              llvm::inconvertibleErrorCode());
        }
        size_t width = enc.chunkSize;
        assert(enc.width % enc.chunkWidth == 0);
        uint64_t size = enc.width / enc.chunkWidth;
        bool isSigned = enc.isSigned;
        std::vector<int64_t> dims{
            (int64_t)size,
        };
        LweSecretKeyID secretKeyID = normKey.index;
        Variance variance = curve.getVariance(1, normKey.dimension, 64);
        CRTDecomposition crt = maybeCrt.value_or(std::vector<int64_t>());
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
            /*.chunkInfo = */
            std::optional<ChunkInfo>(
                {(unsigned int)enc.chunkSize, (unsigned int)enc.chunkWidth}),
        };
      },
      [&](encodings::EncryptedBoolScalarEncoding enc)
          -> llvm::Expected<CircuitGate> {
        auto glweType = type.cast<TFHE::GLWECipherTextType>();
        auto normKey = glweType.getKey().getNormalized().value();
        if ((int)normKey.dimension < curve.minimalLweDimension) {
          return llvm::make_error<llvm::StringError>(
              "Minimal size for security is not attained",
              llvm::inconvertibleErrorCode());
        }
        size_t width =
            mlir::concretelang::FHE::EncryptedBooleanType::getWidth();
        LweSecretKeyID secretKeyID = normKey.index;
        Variance variance = curve.getVariance(1, normKey.dimension, 64);
        return CircuitGate{
            /* .encryption = */ std::optional<EncryptionGate>({
                /* .secretKeyID = */ secretKeyID,
                /* .variance = */ variance,
                /* .encoding = */
                {
                    /* .precision = */ width,
                    /* .crt = */ std::vector<int64_t>(),
                    /* .sign = */ false,
                },
            }),
            /*.shape = */
            {
                /*.width = */ width,
                /*.dimensions = */ std::vector<int64_t>(),
                /*.size = */ 0,
                /*.sign = */ false,
            },
            /*.chunkInfo = */ std::nullopt,
        };
      },
      [&](encodings::PlaintextScalarEncoding enc)
          -> llvm::Expected<CircuitGate> {
        size_t width = type.getIntOrFloatBitWidth();
        bool sign = type.isSignedInteger();
        return CircuitGate{
            /*.encryption = */ std::nullopt,
            /*.shape = */
            {/*.width = */ width,
             /*.dimensions = */ std::vector<int64_t>(),
             /*.size = */ 0,
             /* .sign */ sign},
            /*.chunkInfo = */ std::nullopt,
        };
      },
      [&](encodings::IndexScalarEncoding enc) -> llvm::Expected<CircuitGate> {
        // TODO - The index type is dependant of the target architecture,
        // so actually we assume we target only 64 bits, we need to have
        // some the size of the word of the target system.
        size_t width = 64;
        bool sign = type.isSignedInteger();
        return CircuitGate{
            /*.encryption = */ std::nullopt,
            /*.shape = */
            {/*.width = */ width,
             /*.dimensions = */ std::vector<int64_t>(),
             /*.size = */ 0,
             /* .sign */ sign},
            /*.chunkInfo = */ std::nullopt,
        };
      },
      [&](auto enc) -> llvm::Expected<CircuitGate> {
        return llvm::make_error<llvm::StringError>(
            "cannot convert MLIR type to shape there",
            llvm::inconvertibleErrorCode());
      }};
  auto genericVisitor = overloaded{
      [&](encodings::ScalarEncoding enc) -> llvm::Expected<CircuitGate> {
        return std::visit(scalarVisitor, enc);
      },
      [&](encodings::TensorEncoding enc) -> llvm::Expected<CircuitGate> {
        auto tensor = type.dyn_cast_or_null<mlir::RankedTensorType>();
        auto scalarGate = generateGate(tensor.getElementType(),
                                       enc.scalarEncoding, curve, maybeCrt);
        if (auto err = scalarGate.takeError()) {
          return std::move(err);
        }
        if (maybeCrt.has_value() && scalarGate->isEncrypted()) {
          // When using crt with encrypted tensors, the last dimension of the
          // tensor is for the members of the decomposition. It should not be
          // used.
          scalarGate->shape.dimensions =
              tensor.getShape().take_front(tensor.getShape().size() - 1).vec();
        } else {
          scalarGate->shape.dimensions = tensor.getShape().vec();
        }
        scalarGate->shape.size = 1;
        for (auto dimSize : scalarGate->shape.dimensions) {
          scalarGate->shape.size *= dimSize;
        }
        return scalarGate;
      },
      [&](auto enc) -> llvm::Expected<CircuitGate> {
        return llvm::make_error<llvm::StringError>(
            "cannot convert MLIR type to shape here",
            llvm::inconvertibleErrorCode());
      }};
  return std::visit(genericVisitor, encoding);
}

template <typename V> struct HashValComparator {
  bool operator()(const V &lhs, const V &rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

template <typename V> using Set = llvm::SmallSet<V, 10, HashValComparator<V>>;

void extractCircuitKeys(ClientParameters &output,
                        TFHE::TFHECircuitKeys circuitKeys,
                        concrete::SecurityCurve curve) {

  // Pushing secret keys
  for (auto sk : circuitKeys.secretKeys) {
    clientlib::LweSecretKeyParam skParam;
    skParam.dimension = sk.getNormalized().value().dimension;
    output.secretKeys.push_back(skParam);
  }

  // Pushing keyswitch keys
  for (auto ksk : circuitKeys.keyswitchKeys) {
    clientlib::KeyswitchKeyParam kskParam;
    auto inputNormKey = ksk.getInputKey().getNormalized().value();
    auto outputNormKey = ksk.getOutputKey().getNormalized().value();
    kskParam.inputSecretKeyID = inputNormKey.index;
    kskParam.outputSecretKeyID = outputNormKey.index;
    kskParam.level = ksk.getLevels();
    kskParam.baseLog = ksk.getBaseLog();
    kskParam.variance = curve.getVariance(1, outputNormKey.dimension, 64);
    output.keyswitchKeys.push_back(kskParam);
  }

  // Pushing bootstrap keys
  for (auto bsk : circuitKeys.bootstrapKeys) {
    clientlib::BootstrapKeyParam bskParam;
    auto inputNormKey = bsk.getInputKey().getNormalized().value();
    auto outputNormKey = bsk.getOutputKey().getNormalized().value();
    bskParam.inputSecretKeyID = inputNormKey.index;
    bskParam.outputSecretKeyID = outputNormKey.index;
    bskParam.level = bsk.getLevels();
    bskParam.baseLog = bsk.getBaseLog();
    bskParam.glweDimension = bsk.getGlweDim();
    bskParam.polynomialSize = bsk.getPolySize();
    bskParam.variance =
        curve.getVariance(bsk.getGlweDim(), bsk.getPolySize(), 64);
    bskParam.inputLweDimension = inputNormKey.dimension;
    output.bootstrapKeys.push_back(bskParam);
  }

  // Pushing circuit packing keyswitch keys
  for (auto pksk : circuitKeys.packingKeyswitchKeys) {
    clientlib::PackingKeyswitchKeyParam pkskParam;
    auto inputNormKey = pksk.getInputKey().getNormalized().value();
    auto outputNormKey = pksk.getOutputKey().getNormalized().value();
    pkskParam.inputSecretKeyID = inputNormKey.index;
    pkskParam.outputSecretKeyID = outputNormKey.index;
    pkskParam.level = pksk.getLevels();
    pkskParam.baseLog = pksk.getBaseLog();
    pkskParam.glweDimension = pksk.getGlweDim();
    pkskParam.polynomialSize = pksk.getOutputPolySize();
    pkskParam.inputLweDimension = inputNormKey.dimension;
    pkskParam.variance =
        curve.getVariance(outputNormKey.dimension, outputNormKey.polySize, 64);
    output.packingKeyswitchKeys.push_back(pkskParam);
  }
}

llvm::Expected<std::monostate>
extractCircuitGates(ClientParameters &output, mlir::func::FuncOp funcOp,
                    encodings::CircuitEncodings encodings,
                    concrete::SecurityCurve curve,
                    std::optional<CRTDecomposition> maybeCrt) {

  // Create input and output circuit gate parameters
  auto funcType = funcOp.getFunctionType();

  for (auto val : llvm::zip(funcType.getInputs(), encodings.inputEncodings)) {
    auto ty = std::get<0>(val);
    auto encoding = std::get<1>(val);
    auto gate = generateGate(ty, encoding, curve, maybeCrt);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    output.inputs.push_back(gate.get());
  }
  for (auto val : llvm::zip(funcType.getResults(), encodings.outputEncodings)) {
    auto ty = std::get<0>(val);
    auto encoding = std::get<1>(val);
    auto gate = generateGate(ty, encoding, curve, maybeCrt);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    output.outputs.push_back(gate.get());
  }

  return std::monostate();
}

llvm::Expected<ClientParameters>
createClientParametersFromTFHE(mlir::ModuleOp module,
                               llvm::StringRef functionName, int bitsOfSecurity,
                               encodings::CircuitEncodings encodings,
                               std::optional<CRTDecomposition> maybeCrt) {

  // Check that security curves exist
  const auto curve = concrete::getSecurityCurve(bitsOfSecurity, keyFormat);
  if (curve == nullptr) {
    return StreamStringError("Cannot find security curves for ")
           << bitsOfSecurity << "bits";
  }

  // Check that the specified function can be found
  auto rangeOps = module.getOps<mlir::func::FuncOp>();
  auto funcOp = llvm::find_if(rangeOps, [&](mlir::func::FuncOp op) {
    return op.getName() == functionName;
  });
  if (funcOp == rangeOps.end()) {
    return StreamStringError(
               "cannot find the function for generate client parameters: ")
           << functionName;
  }

  // Create client parameters
  ClientParameters output;
  output.functionName = (std::string)functionName;

  // We extract the keys of the circuit
  auto circuitKeys = TFHE::extractCircuitKeys(module);

  // We extract all the keys used in the circuit
  extractCircuitKeys(output, circuitKeys, *curve);

  // We generate the gates for the inputs aud outputs
  if (auto err =
          extractCircuitGates(output, *funcOp, encodings, *curve, maybeCrt)
              .takeError()) {
    return std::move(err);
  }

  return output;
}

} // namespace concretelang
} // namespace mlir
