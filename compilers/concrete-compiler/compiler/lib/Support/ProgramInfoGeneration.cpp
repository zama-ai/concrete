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

#include "concrete-protocol.pb.h"
#include "concrete/curves.h"
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
#include "concretelang/Common/Values.h"
#include "llvm/Config/abi-breaking.h"

namespace mlir {
namespace concretelang {

const auto keyFormat = concrete::BINARY;
typedef double Variance;

llvm::Expected<concreteprotocol::GateInfo>
generateGate(mlir::Type type, concreteprotocol::EncodingInfo &encoding,
             concrete::SecurityCurve curve) {
  if (!encoding.has_integerciphertext() && !encoding.has_booleanciphertext() &&
      !encoding.has_index() && !encoding.has_plaintext()) {
    return StreamStringError("Tried to generate gate info without encoding.");
  }

  auto output = concreteprotocol::GateInfo();
  if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
    type = tensorType.getElementType();
  }

  if (encoding.has_integerciphertext()) {
    TFHE::GLWESecretKeyNormalized normKey;
    normKey =
        type.cast<TFHE::GLWECipherTextType>().getKey().getNormalized().value();

    auto lweGateInfo = new concreteprotocol::LweCiphertextGateInfo{};
    auto concreteShape = new concreteprotocol::Shape(encoding.shape());
    if (encoding.integerciphertext().has_chunked()) {
      concreteShape->mutable_dimensions()->Add(
          encoding.integerciphertext().chunked().size());
    }
    if (encoding.integerciphertext().has_crt()) {
      concreteShape->mutable_dimensions()->Add(
          encoding.integerciphertext().crt().moduli_size());
    }
    lweGateInfo->set_allocated_concreteshape(concreteShape);
    lweGateInfo->set_integerprecision(64);
    auto encryptionInfo = new concreteprotocol::LweCiphertextEncryptionInfo{};
    encryptionInfo->set_keyid(normKey.index);
    encryptionInfo->set_variance(curve.getVariance(1, normKey.dimension, 64));
    encryptionInfo->set_lwedimension(normKey.dimension);
    auto modulus = new concreteprotocol::Modulus();
    modulus->set_allocated_native(new concreteprotocol::NativeModulus());
    encryptionInfo->set_allocated_modulus(modulus);
    lweGateInfo->set_allocated_encryption(encryptionInfo);
    lweGateInfo->set_compression(concreteprotocol::Compression::none);
    lweGateInfo->set_allocated_integer(new concreteprotocol::IntegerCiphertextEncodingInfo(encoding.integerciphertext()));
    auto rawInfo = new concreteprotocol::RawInfo();
    rawInfo->set_integerprecision(64);
    rawInfo->set_issigned(false);
    auto rawShape = new concreteprotocol::Shape(*concreteShape);
    rawShape->mutable_dimensions()->Add(normKey.dimension+1);
    rawInfo->set_allocated_shape(rawShape);
    output.set_allocated_rawinfo(rawInfo);
    output.set_allocated_lweciphertext(lweGateInfo);
  } else if (encoding.has_booleanciphertext()) {
    auto glweType = type.cast<TFHE::GLWECipherTextType>();
    auto normKey = glweType.getKey().getNormalized().value();
    auto lweGateInfo = new concreteprotocol::LweCiphertextGateInfo{};
    auto concreteShape = new concreteprotocol::Shape(encoding.shape());
    lweGateInfo->set_allocated_concreteshape(concreteShape);
    lweGateInfo->set_integerprecision(64);
    auto encryptionInfo = new concreteprotocol::LweCiphertextEncryptionInfo{};
    encryptionInfo->set_keyid(normKey.index);
    encryptionInfo->set_variance(curve.getVariance(1, normKey.dimension, 64));
    encryptionInfo->set_lwedimension(normKey.dimension);
    auto modulus = new concreteprotocol::Modulus();
    modulus->set_allocated_native(new concreteprotocol::NativeModulus());
    encryptionInfo->set_allocated_modulus(modulus);
    lweGateInfo->set_allocated_encryption(encryptionInfo);
    lweGateInfo->set_compression(concreteprotocol::Compression::none);
    lweGateInfo->set_allocated_boolean(new concreteprotocol::BooleanCiphertextEncodingInfo(encoding.booleanciphertext()));
    auto rawInfo = new concreteprotocol::RawInfo();
    rawInfo->set_integerprecision(64);
    rawInfo->set_issigned(false);
    auto rawShape = new concreteprotocol::Shape(*concreteShape);
    rawShape->mutable_dimensions()->Add(normKey.dimension+1);
    rawInfo->set_allocated_shape(rawShape);
    output.set_allocated_rawinfo(rawInfo);
    output.set_allocated_lweciphertext(lweGateInfo);
  } else if (encoding.has_plaintext()) {
    auto plaintextGateInfo = new concreteprotocol::PlaintextGateInfo{};
    auto shape = new concreteprotocol::Shape(encoding.shape());
    plaintextGateInfo->set_allocated_shape(shape);
    plaintextGateInfo->set_integerprecision(::concretelang::values::getCorrespondingPrecision(type.getIntOrFloatBitWidth()));
    plaintextGateInfo->set_issigned(type.isSignedInteger());
    auto rawInfo = new concreteprotocol::RawInfo();
    rawInfo->set_integerprecision(::concretelang::values::getCorrespondingPrecision(type.getIntOrFloatBitWidth()));
    rawInfo->set_issigned(type.isSignedInteger());
    rawInfo->set_allocated_shape(new concreteprotocol::Shape(*shape));
    output.set_allocated_rawinfo(rawInfo);
    output.set_allocated_plaintext(plaintextGateInfo);
  } else if (encoding.has_index()) {
    // TODO - The index type is dependant of the target architecture,
    // so actually we assume we target only 64 bits, we need to have
    // some the size of the word of the target system.
    auto indexGateInfo = new concreteprotocol::IndexGateInfo{};
    auto shape = new concreteprotocol::Shape(encoding.shape());
    indexGateInfo->set_allocated_shape(shape);
    indexGateInfo->set_integerprecision(64);
    indexGateInfo->set_issigned(type.isSignedInteger());
    auto rawInfo = new concreteprotocol::RawInfo();
    rawInfo->set_integerprecision(64);
    rawInfo->set_issigned(type.isSignedInteger());
    rawInfo->set_allocated_shape(new concreteprotocol::Shape(*shape));
    output.set_allocated_rawinfo(rawInfo);
    output.set_allocated_index(indexGateInfo);
  }
  return output;
}

template <typename V> struct HashValComparator {
  bool operator()(const V &lhs, const V &rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

template <typename V> using Set = llvm::SmallSet<V, 10, HashValComparator<V>>;

concreteprotocol::KeysetInfo
extractCircuitKeys(TFHE::TFHECircuitKeys circuitKeys,
                   concrete::SecurityCurve curve) {

  auto output = concreteprotocol::KeysetInfo();

  // Pushing secret keys
  for (auto sk : circuitKeys.secretKeys) {
    auto info = new concreteprotocol::LweSecretKeyInfo{};
    auto params = new concreteprotocol::LweSecretKeyParams{};
    info->set_id(sk.getNormalized()->index);
    params->set_integerprecision(64);
    params->set_lwedimension(sk.getNormalized().value().dimension);
    params->set_keytype(concreteprotocol::KeyType::binary);
    info->set_allocated_params(params);
    output.mutable_lwesecretkeys()->AddAllocated(info);
  }

  // Pushing keyswitch keys
  for (auto ksk : circuitKeys.keyswitchKeys) {
    auto info = new concreteprotocol::LweKeyswitchKeyInfo{};
    info->set_id(ksk.getIndex());
    info->set_inputid(ksk.getInputKey().getNormalized().value().index);
    info->set_outputid(ksk.getOutputKey().getNormalized().value().index);
    info->set_compression(concreteprotocol::Compression::none);
    auto params = new concreteprotocol::LweKeyswitchKeyParams{};
    params->set_levelcount(ksk.getLevels());
    params->set_baselog(ksk.getBaseLog());
    params->set_variance(curve.getVariance(
        1, ksk.getOutputKey().getNormalized().value().dimension, 64));
    params->set_integerprecision(64);
    auto modulus = new concreteprotocol::Modulus{};
    modulus->set_allocated_native(new concreteprotocol::NativeModulus{});
    params->set_allocated_modulus(modulus);
    params->set_keytype(concreteprotocol::KeyType::binary);
    info->set_allocated_params(params);
    output.mutable_lwekeyswitchkeys()->AddAllocated(info);
  }

  // Pushing bootstrap keys
  for (auto bsk : circuitKeys.bootstrapKeys) {
    auto info = new concreteprotocol::LweBootstrapKeyInfo{};
    info->set_id(bsk.getIndex());
    info->set_inputid(bsk.getInputKey().getNormalized().value().index);
    info->set_outputid(bsk.getOutputKey().getNormalized().value().index);
    info->set_compression(concreteprotocol::Compression::none);
    auto params = new concreteprotocol::LweBootstrapKeyParams{};
    params->set_levelcount(bsk.getLevels());
    params->set_baselog(bsk.getBaseLog());
    params->set_glwedimension(bsk.getGlweDim());
    params->set_polynomialsize(bsk.getPolySize());
    params->set_variance(
        curve.getVariance(bsk.getGlweDim(), bsk.getPolySize(), 64));
    params->set_integerprecision(64);
    auto modulus = new concreteprotocol::Modulus{};
    modulus->set_allocated_native(new concreteprotocol::NativeModulus{});
    params->set_allocated_modulus(modulus);
    params->set_keytype(concreteprotocol::KeyType::binary);
    info->set_allocated_params(params);
    output.mutable_lwebootstrapkeys()->AddAllocated(info);
  }

  // Pushing circuit packing keyswitch keys
  for (auto pksk : circuitKeys.packingKeyswitchKeys) {
    auto info = new concreteprotocol::PackingKeyswitchKeyInfo{};
    info->set_id(pksk.getIndex());
    info->set_inputid(pksk.getInputKey().getNormalized().value().index);
    info->set_outputid(pksk.getOutputKey().getNormalized().value().index);
    info->set_compression(concreteprotocol::Compression::none);
    auto params = new concreteprotocol::PackingKeyswitchKeyParams{};
    params->set_levelcount(pksk.getLevels());
    params->set_baselog(pksk.getBaseLog());
    params->set_glwedimension(pksk.getGlweDim());
    params->set_polynomialsize(pksk.getOutputPolySize());
    params->set_lwedimension(pksk.getInputLweDim());
    params->set_variance(curve.getVariance(
        pksk.getOutputKey().getNormalized().value().dimension,
        pksk.getOutputKey().getNormalized().value().polySize, 64));
    params->set_integerprecision(64);
    auto modulus = new concreteprotocol::Modulus{};
    modulus->set_allocated_native(new concreteprotocol::NativeModulus);
    params->set_allocated_modulus(modulus);
    params->set_keytype(concreteprotocol::KeyType::binary);
    info->set_allocated_params(params);
    output.mutable_packingkeyswitchkeys()->AddAllocated(info);
  }

  return output;
}

llvm::Expected<concreteprotocol::CircuitInfo>
extractCircuitInfo(mlir::ModuleOp module, llvm::StringRef functionName,
                   concreteprotocol::CircuitEncodingInfo &encodings,
                   concrete::SecurityCurve curve) {

  auto output = concreteprotocol::CircuitInfo();

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
  output.set_name(functionName.str());

  // Create input and output circuit gate parameters
  auto funcType = (*funcOp).getFunctionType();
  for (unsigned int i = 0; i < funcType.getNumInputs(); i++) {
    auto ty = funcType.getInput(i);
    auto encoding = encodings.inputs(i);
    auto maybeGate = generateGate(ty, encoding, curve);
    if (!maybeGate) {
      return maybeGate.takeError();
    }
    output.mutable_inputs()->AddAllocated(
        new concreteprotocol::GateInfo(*maybeGate));
  }
  for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
    auto ty = funcType.getResult(i);
    auto encoding = encodings.outputs(i);
    auto maybeGate = generateGate(ty, encoding, curve);
    if (!maybeGate) {
      return maybeGate.takeError();
    }
    output.mutable_outputs()->AddAllocated(
        new concreteprotocol::GateInfo(*maybeGate));
  }

  return output;
}

llvm::Expected<concreteprotocol::ProgramInfo>
createProgramInfoFromTFHE(mlir::ModuleOp module, llvm::StringRef functionName,
                          int bitsOfSecurity,
                          concreteprotocol::CircuitEncodingInfo &encodings) {

  // Check that security curves exist
  const auto curve = concrete::getSecurityCurve(bitsOfSecurity, keyFormat);
  if (curve == nullptr) {
    return StreamStringError("Cannot find security curves for ")
           << bitsOfSecurity << "bits";
  }

  // Create Program Info
  auto output = concreteprotocol::ProgramInfo();

  // We extract the keys of the circuit
  auto keysetInfo =
      extractCircuitKeys(TFHE::extractCircuitKeys(module), *curve);
  output.set_allocated_keyset(new concreteprotocol::KeysetInfo(keysetInfo));

  // We generate the gates for the inputs aud outputs
  auto maybeCircuitInfo =
      extractCircuitInfo(module, functionName, encodings, *curve);
  if (!maybeCircuitInfo) {
    return maybeCircuitInfo.takeError();
  }
  output.mutable_circuits()->AddAllocated(
      new concreteprotocol::CircuitInfo(*maybeCircuitInfo));

  return output;
}

} // namespace concretelang
} // namespace mlir
