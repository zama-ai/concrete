// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.
#include <cassert>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <map>
#include <memory>
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
#include "concrete-protocol.pb.h"

namespace mlir {
namespace concretelang {

namespace clientlib = ::concretelang::clientlib;
namespace protocol = concreteprotocol;
using ::concretelang::clientlib::ChunkInfo;
using ::concretelang::clientlib::CircuitGate;
using ::concretelang::clientlib::ClientParameters;
using ::concretelang::clientlib::Encoding;
using ::concretelang::clientlib::EncryptionGate;
using ::concretelang::clientlib::LweSecretKeyID;
using ::concretelang::clientlib::Precision;
using ::concretelang::clientlib::Variance;

const auto keyFormat = concrete::BINARY;

llvm::Expected<std::unique_ptr<protocol::GateInfo>>
generateGate(mlir::Type type, std::unique_ptr<protocol::EncodingInfo> encoding, concrete::SecurityCurve curve) {
  auto output = std::make_unique<protocol::GateInfo>();
  if (encoding->has_integerciphertext()){
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
        Variance variance = curve.getVariance(1, normKey.dimension, 64);

        output->set_allocated_shape(encoding->release_shape());
        auto encryptionInfo = new protocol::LweCiphertextEncryptionInfo{};
        encryptionInfo->set_keyid(normKey.index);
        encryptionInfo->set_variance(variance);
        encryptionInfo->set_integerprecision(64);
        auto gateInfo = new protocol::LweCiphertextGateInfo{};
        gateInfo->set_allocated_encryption(encryptionInfo);
        gateInfo->set_allocated_integer(encoding->release_integerciphertext());
        output->set_allocated_lweciphertext(gateInfo);
        return output;
  } else if (encoding->has_booleanciphertext()){
        auto glweType = type.cast<TFHE::GLWECipherTextType>();
        auto normKey = glweType.getKey().getNormalized().value();
        Variance variance = curve.getVariance(1, normKey.dimension, 64);
        output->set_allocated_shape(encoding->release_shape());
        auto encryptionInfo = new protocol::LweCiphertextEncryptionInfo{};
        encryptionInfo->set_keyid(normKey.index);
        encryptionInfo->set_variance(variance);
        encryptionInfo->set_integerprecision(64);
        auto gateInfo = new protocol::LweCiphertextGateInfo{};
        gateInfo->set_allocated_encryption(encryptionInfo);
        gateInfo->set_allocated_boolean(encoding->release_booleanciphertext());
        output->set_allocated_lweciphertext(gateInfo);
        return output;
  } else if (encoding->has_plaintext()){
        auto gateInfo = new protocol::PlaintextGateInfo{};
        gateInfo->set_issigned(type.isSignedInteger());
        gateInfo->set_integerprecision(type.getIntOrFloatBitWidth());
        output->set_allocated_plaintext(gateInfo);
        return output;
  } else if (encoding->has_index()){
        // TODO - The index type is dependant of the target architecture,
        // so actually we assume we target only 64 bits, we need to have
        // some the size of the word of the target system.
        auto gateInfo = new protocol::IndexGateInfo{};
        gateInfo->set_issigned(type.isSignedInteger());
        gateInfo->set_integerprecision(64);
        output->set_allocated_index(gateInfo);
        return output;
  }
  return StreamStringError("Tried to generate gate info without encoding.");
}

template <typename V> struct HashValComparator {
  bool operator()(const V &lhs, const V &rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

template <typename V> using Set = llvm::SmallSet<V, 10, HashValComparator<V>>;

std::unique_ptr<protocol::KeysetInfo> extractCircuitKeys(
  TFHE::TFHECircuitKeys circuitKeys,
  concrete::SecurityCurve curve) {

  auto output = std::make_unique<protocol::KeysetInfo>();

  // Pushing secret keys
  for (auto sk : circuitKeys.secretKeys) {
    auto info = new protocol::LweSecretKeyInfo{};
    auto params = new protocol::LweSecretKeyParams{};
    info->set_id(sk.getNormalized()->index);
    params->set_integerprecision(64);
    params->set_lwedimension(sk.getNormalized().value().dimension);
    params->set_keytype(protocol::KeyType::binary);
    info->set_allocated_parameters(params);
    output->mutable_lwesecretkeys()->AddAllocated(info);
  }

  // Pushing keyswitch keys
  for (auto ksk : circuitKeys.keyswitchKeys) {
    auto info = new protocol::LweKeyswitchKeyInfo{};
    info->set_id(ksk.getIndex());
    info->set_inputid(ksk.getInputKey().getNormalized().value().index);
    info->set_outputid(ksk.getOutputKey().getNormalized().value().index);
    info->set_compression(protocol::LweKeyswitchKeyInfo_Compression_none);
    auto params = new protocol::LweKeyswitchKeyParams{};
    params->set_levelcount(ksk.getLevels());
    params->set_baselog(ksk.getBaseLog());
    params->set_variance(curve.getVariance(1, ksk.getOutputKey().getNormalized().value().dimension, 64));
    params->set_integerprecision(64);
    auto modulus = new protocol::Modulus{};
    modulus->set_allocated_native(new protocol::NativeModulus{});
    params->set_allocated_modulus(modulus);
    params->set_keytype(protocol::KeyType::binary);
    info->set_allocated_params(params);
    output->mutable_lwekeyswitchkeys()->AddAllocated(info);
  }

  // Pushing bootstrap keys
  for (auto bsk : circuitKeys.bootstrapKeys) {
    auto info = new protocol::LweBootstrapKeyInfo{};
    info->set_id(bsk.getIndex());
    info->set_inputid(bsk.getInputKey().getNormalized().value().index);
    info->set_outputid(bsk.getOutputKey().getNormalized().value().index);
    info->set_compression(protocol::LweBootstrapKeyInfo_Compression_none);
    auto params = new protocol::LweBootstrapKeyParams{};
    params->set_levelcount(bsk.getLevels());
    params->set_baselog(bsk.getBaseLog());
    params->set_glwedimension(bsk.getGlweDim());
    params->set_polynomialsize(bsk.getPolySize());
    params->set_variance(curve.getVariance(bsk.getGlweDim(), bsk.getPolySize(), 64));
    params->set_integerprecision(64);
    auto modulus = new protocol::Modulus{};
    modulus->set_allocated_native(new protocol::NativeModulus{});
    params->set_allocated_modulus(modulus);
    params->set_keytype(protocol::KeyType::binary);
    info->set_allocated_params(params);
    output->mutable_lwebootstrapkeys()->AddAllocated(info);
  }

  // Pushing circuit packing keyswitch keys
  for (auto pksk : circuitKeys.packingKeyswitchKeys) {
    auto info = new protocol::PackingKeyswitchKeyInfo{};
    info->set_id(pksk.getIndex());
    info->set_inputid(pksk.getInputKey().getNormalized().value().index);
    info->set_outputid(pksk.getOutputKey().getNormalized().value().index);
    info->set_compression(protocol::PackingKeyswitchKeyInfo_Compression_none);
    auto params = new protocol::PackingKeyswitchKeyParams{};
    params->set_levelcount(pksk.getLevels());
    params->set_baselog(pksk.getBaseLog());
    params->set_glwedimension(pksk.getGlweDim());
    params->set_polynomialsize(pksk.getOutputPolySize());
    params->set_lwedimension(pksk.getInputLweDim());
    params->set_variance(curve.getVariance(pksk.getOutputKey().getNormalized().value().dimension, pksk.getOutputKey().getNormalized().value().polySize, 64));
    params->set_integerprecision(64);
    auto modulus = new protocol::Modulus{};
    modulus->set_allocated_native(new protocol::NativeModulus);
    params->set_allocated_modulus(modulus);
    params->set_keytype(protocol::KeyType::binary);
    info->set_allocated_params(params);
    output->mutable_packingkeyswitchkeys()->AddAllocated(info);
  }

  return output;
}

llvm::Expected<std::unique_ptr<protocol::CircuitInfo>>
extractCircuitInfo(mlir::ModuleOp module,
                   llvm::StringRef functionName,
                   encodings::CircuitEncodings encodings,
                   concrete::SecurityCurve curve) {
  
  auto output = std::make_unique<protocol::CircuitInfo>();

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
  output->set_name(functionName.str());

  // Create input and output circuit gate parameters
  auto funcType = (*funcOp).getFunctionType();
  for (auto val : llvm::zip(funcType.getInputs(), encodings.inputEncodings)) {
    auto ty = std::get<0>(val);
    auto encoding = std::unique_ptr<protocol::EncodingInfo>(std::get<1>(val).release());
    auto maybeGate = generateGate(ty, std::move(encoding), curve);
    if (!maybeGate){
      return std::move(maybeGate.takeError());
    }
    output->mutable_inputs()->AddAllocated((*maybeGate).release());
  }
  for (auto val : llvm::zip(funcType.getResults(), encodings.outputEncodings)) {
    auto ty = std::get<0>(val);
    auto encoding = std::unique_ptr<protocol::EncodingInfo>(std::get<1>(val).release());
    auto maybeGate = generateGate(ty, std::move(encoding), curve);
    if (!maybeGate){
      return std::move(maybeGate.takeError());
    }
    output->mutable_outputs()->AddAllocated((*maybeGate).release());
  }

  return output;
}

llvm::Expected<std::unique_ptr<protocol::ProgramInfo>>
createClientParametersFromTFHE(mlir::ModuleOp module,
                               llvm::StringRef functionName, 
                               int bitsOfSecurity,
                               encodings::CircuitEncodings encodings) {

  // Check that security curves exist
  const auto curve = concrete::getSecurityCurve(bitsOfSecurity, keyFormat);
  if (curve == nullptr) {
    return StreamStringError("Cannot find security curves for ")
           << bitsOfSecurity << "bits";
  }

  // Create Program Info
  auto output = std::make_unique<protocol::ProgramInfo>();

  // We extract the keys of the circuit
  auto keysetInfo = extractCircuitKeys(TFHE::extractCircuitKeys(module), *curve);
  output->set_allocated_keyset(keysetInfo.release());

  // We generate the gates for the inputs aud outputs
  auto maybeCircuitInfo = extractCircuitInfo(module, functionName, encodings, *curve);
  if (!maybeCircuitInfo){
    return std::move(maybeCircuitInfo.takeError());
  }
  output->mutable_circuits()->AddAllocated((*maybeCircuitInfo).release());

  return output;
}

} // namespace concretelang
} // namespace mlir
