// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <unordered_set>
#include <variant>

#include "capnp/message.h"
#include "concrete-cpu.h"
#include "concrete-protocol.capnp.h"
#include "concrete/curves.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Values.h"
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/Support/Error.h"

using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {

const auto keyFormat = concrete::BINARY;
typedef double Variance;

llvm::Expected<Message<concreteprotocol::GateInfo>>
generateGate(mlir::Type inputType,
             const Message<concreteprotocol::EncodingInfo> &inputEncodingInfo,
             concrete::SecurityCurve curve,
             concreteprotocol::Compression compression) {

  auto inputEncoding = inputEncodingInfo.asReader().getEncoding();
  if (!inputEncoding.hasIntegerCiphertext() &&
      !inputEncoding.hasBooleanCiphertext() && !inputEncoding.hasIndex() &&
      !inputEncoding.hasPlaintext()) {
    return StreamStringError("Tried to generate gate info without encoding.");
  }
  auto inputShape = inputEncodingInfo.asReader().getShape();
  if (auto inputTensorType = inputType.dyn_cast<mlir::RankedTensorType>()) {
    inputType = inputTensorType.getElementType();
  }
  auto output = Message<concreteprotocol::GateInfo>();

  if (inputEncoding.hasIntegerCiphertext()) {
    auto normKey = inputType.cast<TFHE::GLWECipherTextType>()
                       .getKey()
                       .getNormalized()
                       .value();
    auto lweCiphertextGateInfo =
        output.asBuilder().initTypeInfo().initLweCiphertext();
    auto concreteShape = lweCiphertextGateInfo.initConcreteShape();
    lweCiphertextGateInfo.setAbstractShape(inputShape);
    auto encodingDimensions = inputShape.getDimensions();
    size_t gateDimensionsSize = inputShape.getDimensions().size() + 1;
    if (inputEncoding.getIntegerCiphertext().getMode().hasChunked() ||
        inputEncoding.getIntegerCiphertext().getMode().hasCrt()) {
      gateDimensionsSize++;
    }
    auto gateDimensions = concreteShape.initDimensions(gateDimensionsSize);
    for (size_t i = 0; i < encodingDimensions.size(); i++) {
      gateDimensions.set(i, encodingDimensions[i]);
    }
    if (inputEncoding.getIntegerCiphertext().getMode().hasChunked()) {
      gateDimensions.set(encodingDimensions.size(),
                         inputEncoding.getIntegerCiphertext()
                             .getMode()
                             .getChunked()
                             .getSize());
    }
    if (inputEncoding.getIntegerCiphertext().getMode().hasCrt()) {
      gateDimensions.set(encodingDimensions.size(),
                         inputEncoding.getIntegerCiphertext()
                             .getMode()
                             .getCrt()
                             .getModuli()
                             .size());
    }
    auto ciphertextSize = normKey.dimension + 1;
    if (compression == concreteprotocol::Compression::SEED) {
      ciphertextSize = 3;
    }
    gateDimensions.set(gateDimensionsSize - 1, ciphertextSize);
    lweCiphertextGateInfo.setIntegerPrecision(64);
    auto encryptionInfo = lweCiphertextGateInfo.initEncryption();
    encryptionInfo.setKeyId(normKey.index);
    encryptionInfo.setVariance(curve.getVariance(1, normKey.dimension, 64));
    encryptionInfo.setLweDimension(normKey.dimension);
    encryptionInfo.initModulus().initMod().initNative();
    lweCiphertextGateInfo.setCompression(compression);
    lweCiphertextGateInfo.initEncoding().setInteger(
        inputEncoding.getIntegerCiphertext());
    auto rawInfo = output.asBuilder().initRawInfo();
    auto rawShape = rawInfo.initShape();
    rawShape.setDimensions(gateDimensions.asReader());
    rawInfo.setIntegerPrecision(64);
    rawInfo.setIsSigned(false);
  } else if (inputEncoding.hasBooleanCiphertext()) {
    auto glweType = inputType.cast<TFHE::GLWECipherTextType>();
    auto normKey = glweType.getKey().getNormalized().value();
    auto lweCiphertextGateInfo =
        output.asBuilder().initTypeInfo().initLweCiphertext();
    auto encodingDimensions = inputShape.getDimensions();
    size_t gateDimensionsSize = inputShape.getDimensions().size() + 1;
    lweCiphertextGateInfo.setAbstractShape(inputShape);
    auto gateDimensions =
        lweCiphertextGateInfo.initConcreteShape().initDimensions(
            gateDimensionsSize);
    for (size_t i = 0; i < encodingDimensions.size(); i++) {
      gateDimensions.set(i, encodingDimensions[i]);
    }
    auto ciphertextSize = normKey.dimension + 1;
    if (compression == concreteprotocol::Compression::SEED) {
      ciphertextSize = 3;
    }
    gateDimensions.set(gateDimensionsSize - 1, ciphertextSize);
    lweCiphertextGateInfo.setIntegerPrecision(64);
    auto encryptionInfo = lweCiphertextGateInfo.initEncryption();
    encryptionInfo.setKeyId(normKey.index);
    encryptionInfo.setVariance(curve.getVariance(1, normKey.dimension, 64));
    encryptionInfo.setLweDimension(normKey.dimension);
    encryptionInfo.initModulus().initMod().initNative();
    lweCiphertextGateInfo.setCompression(compression);
    lweCiphertextGateInfo.initEncoding().initBoolean();

    auto rawInfo = output.asBuilder().initRawInfo();
    auto rawShape = rawInfo.initShape();
    rawShape.setDimensions(gateDimensions.asReader());
    rawInfo.setIntegerPrecision(64);
    rawInfo.setIsSigned(false);
  } else if (inputEncoding.hasPlaintext()) {
    auto plaintextGateInfo = output.asBuilder().initTypeInfo().initPlaintext();
    plaintextGateInfo.setShape(inputShape);
    plaintextGateInfo.setIntegerPrecision(
        ::concretelang::values::getCorrespondingPrecision(
            inputType.getIntOrFloatBitWidth()));
    plaintextGateInfo.setIsSigned(inputType.isSignedInteger());

    auto rawInfo = output.asBuilder().initRawInfo();
    rawInfo.setShape(inputShape);
    rawInfo.setIntegerPrecision(
        ::concretelang::values::getCorrespondingPrecision(
            inputType.getIntOrFloatBitWidth()));
    rawInfo.setIsSigned(inputType.isSignedInteger());
  } else if (inputEncoding.hasIndex()) {
    // TODO - The index type is dependant of the target architecture,
    // so actually we assume we target only 64 bits, we need to have
    // some the size of the word of the target system.
    auto indexGateInfo = output.asBuilder().initTypeInfo().initIndex();
    indexGateInfo.setShape(inputShape);
    indexGateInfo.setIntegerPrecision(64);
    indexGateInfo.setIsSigned(inputType.isSignedInteger());

    auto rawInfo = output.asBuilder().initRawInfo();
    rawInfo.setShape(inputShape);
    rawInfo.setIntegerPrecision(64);
    rawInfo.setIsSigned(inputType.isSignedInteger());
  }
  return output;
}

Message<concreteprotocol::KeysetInfo>
extractKeysetInfo(TFHE::TFHECircuitKeys circuitKeys,
                  concrete::SecurityCurve curve, bool compressEvaluationKeys,
                  concrete_optimizer::PublicKey withPublicKeys) {
  auto output = Message<concreteprotocol::KeysetInfo>();

  // Pushing secret keys and public keys if requested
  auto secretKeysBuilder =
      output.asBuilder().initLweSecretKeys(circuitKeys.secretKeys.size());
  for (size_t i = 0; i < circuitKeys.secretKeys.size(); i++) {
    auto infoMessage = Message<concreteprotocol::LweSecretKeyInfo>();
    auto sk = circuitKeys.secretKeys[i];
    auto skIndex = sk.getNormalized()->index;
    auto lweDimension = sk.getNormalized().value().dimension;
    infoMessage.asBuilder().setId(skIndex);
    auto paramsBuilder = infoMessage.asBuilder().initParams();
    paramsBuilder.setIntegerPrecision(64);
    paramsBuilder.setLweDimension(lweDimension);
    paramsBuilder.setKeyType(concreteprotocol::KeyType::BINARY);
    secretKeysBuilder.setWithCaveats(i, infoMessage.asReader());
  }

  // Pushing keyswitch keys
  auto keyswitchKeysBuilder =
      output.asBuilder().initLweKeyswitchKeys(circuitKeys.keyswitchKeys.size());
  for (size_t i = 0; i < circuitKeys.keyswitchKeys.size(); i++) {
    auto infoMessage = Message<concreteprotocol::LweKeyswitchKeyInfo>();
    auto ksk = circuitKeys.keyswitchKeys[i];
    infoMessage.asBuilder().setId(ksk.getIndex());
    infoMessage.asBuilder().setInputId(
        ksk.getInputKey().getNormalized().value().index);
    infoMessage.asBuilder().setOutputId(
        ksk.getOutputKey().getNormalized().value().index);
    if (!compressEvaluationKeys) {
      infoMessage.asBuilder().setCompression(
          concreteprotocol::Compression::NONE);
    } else {
      infoMessage.asBuilder().setCompression(
          concreteprotocol::Compression::SEED);
    }
    auto paramsBuilder = infoMessage.asBuilder().initParams();
    paramsBuilder.setLevelCount(ksk.getLevels());
    paramsBuilder.setBaseLog(ksk.getBaseLog());
    paramsBuilder.setVariance(curve.getVariance(
        1, ksk.getOutputKey().getNormalized().value().dimension, 64));
    paramsBuilder.setIntegerPrecision(64);
    paramsBuilder.setInputLweDimension(
        ksk.getInputKey().getNormalized().value().dimension);
    paramsBuilder.setOutputLweDimension(
        ksk.getOutputKey().getNormalized().value().dimension);
    paramsBuilder.setKeyType(concreteprotocol::KeyType::BINARY);
    paramsBuilder.initModulus().initMod().initNative();
    keyswitchKeysBuilder.setWithCaveats(i, infoMessage.asReader());
  }

  // Pushing bootstrap keys
  auto bootstrapKeysBuilder =
      output.asBuilder().initLweBootstrapKeys(circuitKeys.bootstrapKeys.size());
  for (size_t i = 0; i < circuitKeys.bootstrapKeys.size(); i++) {
    auto infoMessage = Message<concreteprotocol::LweBootstrapKeyInfo>();
    auto bsk = circuitKeys.bootstrapKeys[i];
    infoMessage.asBuilder().setId(bsk.getIndex());
    infoMessage.asBuilder().setInputId(
        bsk.getInputKey().getNormalized().value().index);
    infoMessage.asBuilder().setOutputId(
        bsk.getOutputKey().getNormalized().value().index);
    if (!compressEvaluationKeys) {
      infoMessage.asBuilder().setCompression(
          concreteprotocol::Compression::NONE);
    } else {
      infoMessage.asBuilder().setCompression(
          concreteprotocol::Compression::SEED);
    }
    auto paramsBuilder = infoMessage.asBuilder().initParams();
    paramsBuilder.setLevelCount(bsk.getLevels());
    paramsBuilder.setBaseLog(bsk.getBaseLog());
    paramsBuilder.setGlweDimension(bsk.getGlweDim());
    paramsBuilder.setPolynomialSize(bsk.getPolySize());
    paramsBuilder.setInputLweDimension(
        bsk.getInputKey().getNormalized().value().dimension);
    paramsBuilder.setVariance(
        curve.getVariance(bsk.getGlweDim(), bsk.getPolySize(), 64));
    paramsBuilder.setIntegerPrecision(64);
    paramsBuilder.setKeyType(concreteprotocol::KeyType::BINARY);
    paramsBuilder.initModulus().initMod().initNative();
    bootstrapKeysBuilder.setWithCaveats(i, infoMessage.asReader());
  }

  // Pushing circuit packing keyswitch keys
  auto packingKeyswitchKeysBuilder =
      output.asBuilder().initPackingKeyswitchKeys(
          circuitKeys.packingKeyswitchKeys.size());
  for (size_t i = 0; i < circuitKeys.packingKeyswitchKeys.size(); i++) {
    auto infoMessage = Message<concreteprotocol::PackingKeyswitchKeyInfo>();
    auto pksk = circuitKeys.packingKeyswitchKeys[i];
    infoMessage.asBuilder().setId(pksk.getIndex());
    infoMessage.asBuilder().setInputId(
        pksk.getInputKey().getNormalized().value().index);
    infoMessage.asBuilder().setOutputId(
        pksk.getOutputKey().getNormalized().value().index);
    if (!compressEvaluationKeys) {
      infoMessage.asBuilder().setCompression(
          concreteprotocol::Compression::NONE);
    } else {
      infoMessage.asBuilder().setCompression(
          concreteprotocol::Compression::SEED);
    }
    auto paramsBuilder = infoMessage.asBuilder().initParams();
    paramsBuilder.setLevelCount(pksk.getLevels());
    paramsBuilder.setBaseLog(pksk.getBaseLog());
    paramsBuilder.setGlweDimension(pksk.getGlweDim());
    paramsBuilder.setPolynomialSize(pksk.getOutputPolySize());
    paramsBuilder.setInputLweDimension(
        pksk.getInputKey().getNormalized().value().dimension);
    paramsBuilder.setInnerLweDimension(pksk.getInnerLweDim());
    paramsBuilder.setVariance(curve.getVariance(
        pksk.getOutputKey().getNormalized().value().dimension,
        pksk.getOutputKey().getNormalized().value().polySize, 64));
    paramsBuilder.setIntegerPrecision(64);
    paramsBuilder.setKeyType(concreteprotocol::KeyType::BINARY);
    paramsBuilder.initModulus().initMod().initNative();
    packingKeyswitchKeysBuilder.setWithCaveats(i, infoMessage.asReader());
  }

  if (withPublicKeys == concrete_optimizer::PublicKey::None)
    return output;
  auto publicKeysBuilder =
      output.asBuilder().initLwePublicKeys(circuitKeys.inputKeys.size());
  for (auto e : llvm::enumerate(circuitKeys.inputKeys)) {
    Message<concreteprotocol::LwePublicKeyInfo> publicKeyInfo;
    auto publicKeyInfoBuilder = publicKeyInfo.asBuilder();
    auto inputKey = e.value();
    auto skIndex = inputKey.getNormalized()->index;
    auto lweDimension = inputKey.getNormalized().value().dimension;
    publicKeyInfoBuilder.setId(skIndex);
    auto publicKeyParams = publicKeyInfoBuilder.initParams();
    auto encryptionVariance = curve.getVariance(1, lweDimension, 64);
    if (withPublicKeys == concrete_optimizer::PublicKey::Classic) {
      auto classicParams = publicKeyParams.initClassic();
      classicParams.setIntegerPrecision(64);
      classicParams.setLweDimension(lweDimension);
      classicParams.setVariance(encryptionVariance);
      classicParams.setZeroEncryptionCount(
          concrete_cpu_public_key_zero_encryption_count(lweDimension));
    } else if (withPublicKeys == concrete_optimizer::PublicKey::Compact) {
      auto compactParams = publicKeyParams.initCompact();
      compactParams.setIntegerPrecision(64);
      compactParams.setLweDimension(lweDimension);
      compactParams.setVariance(encryptionVariance);
    }
    publicKeysBuilder.setWithCaveats(e.index(), publicKeyInfo.asReader());
  }

  return output;
}

llvm::Expected<Message<concreteprotocol::CircuitInfo>>
extractCircuitInfo(mlir::func::FuncOp funcOp,
                   concreteprotocol::CircuitEncodingInfo::Reader encodings,
                   concrete::SecurityCurve curve,
                   bool compressInputCiphertexts) {

  auto output = Message<concreteprotocol::CircuitInfo>();

  // Create input and output circuit gate parameters
  auto funcType = funcOp.getFunctionType();

  output.asBuilder().setName(encodings.getName().cStr());
  output.asBuilder().initInputs(funcType.getNumInputs());
  output.asBuilder().initOutputs(funcType.getNumResults());

  for (unsigned int i = 0; i < funcType.getNumInputs(); i++) {
    auto ty = funcType.getInput(i);
    auto encoding = encodings.getInputs()[i];
    auto compression = compressInputCiphertexts
                           ? concreteprotocol::Compression::SEED
                           : concreteprotocol::Compression::NONE;
    auto maybeGate = generateGate(ty, encoding, curve, compression);
    if (!maybeGate) {
      return maybeGate.takeError();
    }
    output.asBuilder().getInputs().setWithCaveats(i, maybeGate->asReader());
  }
  for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
    auto ty = funcType.getResult(i);
    auto encoding = encodings.getOutputs()[i];
    auto compression = concreteprotocol::Compression::NONE;
    auto maybeGate = generateGate(ty, encoding, curve, compression);
    if (!maybeGate) {
      return maybeGate.takeError();
    }
    output.asBuilder().getOutputs().setWithCaveats(i, maybeGate->asReader());
  }

  return output;
}

llvm::Expected<Message<concreteprotocol::ProgramInfo>> extractProgramInfo(
    mlir::ModuleOp module,
    const Message<concreteprotocol::ProgramEncodingInfo> &encodings,
    concrete::SecurityCurve curve, bool compressInputCiphertexts) {

  auto output = Message<concreteprotocol::ProgramInfo>();
  auto circuitsCount = encodings.asReader().getCircuits().size();
  auto circuitsBuilder = output.asBuilder().initCircuits(circuitsCount);
  auto rangeOps = module.getOps<mlir::func::FuncOp>();

  for (size_t i = 0; i < circuitsCount; i++) {
    auto circuitEncoding = encodings.asReader().getCircuits()[i];
    auto functionName = circuitEncoding.getName();
    auto funcOp = llvm::find_if(rangeOps, [&](mlir::func::FuncOp op) {
      return op.getName() == functionName.cStr();
    });
    if (funcOp == rangeOps.end()) {
      return StreamStringError("cannot find the following function to generate "
                               "program info: ")
             << functionName.cStr();
    }

    auto maybeCircuitInfo = extractCircuitInfo(*funcOp, circuitEncoding, curve,
                                               compressInputCiphertexts);
    if (!maybeCircuitInfo) {
      return maybeCircuitInfo.takeError();
    }

    circuitsBuilder.setWithCaveats(i, (*maybeCircuitInfo).asReader());
  }

  return output;
}

llvm::Expected<Message<concreteprotocol::ProgramInfo>>
createProgramInfoFromTfheDialect(
    mlir::ModuleOp module, int bitsOfSecurity,
    const Message<concreteprotocol::ProgramEncodingInfo> &encodings,
    bool compressEvaluationKeys, bool compressInputCiphertexts,
    concrete_optimizer::PublicKey withPublicKeys) {

  // Check that security curves exist
  const auto curve = concrete::getSecurityCurve(bitsOfSecurity, keyFormat);
  if (curve == nullptr) {
    return StreamStringError("Cannot find security curves for ")
           << bitsOfSecurity << "bits";
  }

  // We generate the circuit infos from the module.
  auto maybeProgramInfo =
      extractProgramInfo(module, encodings, *curve, compressInputCiphertexts);
  if (!maybeProgramInfo) {
    return maybeProgramInfo.takeError();
  }

  // Extract the output Program Info.
  Message<concreteprotocol::ProgramInfo> output = *maybeProgramInfo;

  // We extract the keys of the circuit
  auto keysetInfo = extractKeysetInfo(TFHE::extractCircuitKeys(module), *curve,
                                      compressEvaluationKeys, withPublicKeys);
  output.asBuilder().setKeyset(keysetInfo.asReader());

  return output;
}

} // namespace concretelang
} // namespace mlir
