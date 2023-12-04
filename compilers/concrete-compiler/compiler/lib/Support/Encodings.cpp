// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Support/Encodings.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/Utils.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/Support/Variants.h"
#include "kj/common.h"
#include <functional>
#include <memory>
#include <optional>
#include <variant>

namespace FHE = mlir::concretelang::FHE;
using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {
namespace encodings {

llvm::Expected<Message<concreteprotocol::EncodingInfo>>
encodingFromType(mlir::Type ty) {

  if (auto eintTy = ty.dyn_cast<FHE::FheIntegerInterface>()) {
    auto output = Message<concreteprotocol::EncodingInfo>();
    auto encodingBuilder =
        output.asBuilder().getEncoding().initIntegerCiphertext();
    encodingBuilder.setIsSigned(eintTy.isSigned());
    encodingBuilder.setWidth(eintTy.getWidth());
    output.asBuilder().getShape().initDimensions(0);
    return std::move(output);
  } else if (auto eboolTy = ty.dyn_cast<FHE::EncryptedBooleanType>()) {
    auto output = Message<concreteprotocol::EncodingInfo>();
    output.asBuilder().getEncoding().initBooleanCiphertext();
    output.asBuilder().getShape().initDimensions(0);
    return std::move(output);
  } else if (auto intTy = ty.dyn_cast<mlir::IntegerType>()) {
    auto output = Message<concreteprotocol::EncodingInfo>();
    output.asBuilder().getEncoding().initPlaintext();
    output.asBuilder().getShape().initDimensions(0);
    return std::move(output);
  } else if (auto indexTy = ty.dyn_cast<mlir::IndexType>()) {
    auto output = Message<concreteprotocol::EncodingInfo>();
    output.asBuilder().getEncoding().initIndex();
    output.asBuilder().getShape().initDimensions(0);
    return std::move(output);
  } else if (auto tensorTy = ty.dyn_cast<mlir::RankedTensorType>()) {
    auto maybeElementEncoding = encodingFromType(tensorTy.getElementType());
    if (!maybeElementEncoding) {
      return maybeElementEncoding.takeError();
    }
    auto output = std::move(*maybeElementEncoding);
    auto shapeBuilder =
        output.asBuilder().initShape().initDimensions(tensorTy.getRank());
    for (int64_t i = 0; i < tensorTy.getRank(); i++) {
      shapeBuilder.set(i, tensorTy.getShape()[i]);
    }
    return std::move(output);
  }
  return StreamStringError("Failed to recognize encoding for type : ") << ty;
}

llvm::Expected<Message<concreteprotocol::CircuitEncodingInfo>>
getCircuitEncodings(llvm::StringRef functionName, mlir::ModuleOp module) {
  // Find the input function
  auto rangeOps = module.getOps<mlir::func::FuncOp>();
  auto funcOp = llvm::find_if(rangeOps, [&](mlir::func::FuncOp op) {
    return op.getName() == functionName;
  });
  if (funcOp == rangeOps.end()) {
    return StreamStringError("Function not found, name='")
           << functionName << "', cannot get circuit encodings";
  }
  auto funcType = (*funcOp).getFunctionType();

  // Retrieve input/output encodings
  auto circuitEncodings = Message<concreteprotocol::CircuitEncodingInfo>();
  auto inputsBuilder =
      circuitEncodings.asBuilder().initInputs(funcType.getNumInputs());
  for (size_t i = 0; i < funcType.getNumInputs(); i++) {
    auto ty = funcType.getInputs()[i];
    auto maybeEncoding = encodingFromType(ty);
    if (!maybeEncoding) {
      return maybeEncoding.takeError();
    }
    inputsBuilder.setWithCaveats(i, maybeEncoding->asReader());
  }
  auto outputsBuilder =
      circuitEncodings.asBuilder().initOutputs(funcType.getNumResults());
  for (size_t i = 0; i < funcType.getNumResults(); i++) {
    auto ty = funcType.getResults()[i];
    auto maybeEncoding = encodingFromType(ty);
    if (!maybeEncoding) {
      return maybeEncoding.takeError();
    }
    outputsBuilder.setWithCaveats(i, maybeEncoding->asReader());
  }

  return std::move(circuitEncodings);
}

void setCircuitEncodingModes(
    Message<concreteprotocol::CircuitEncodingInfo> &info,
    std::optional<
        Message<concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode>>
        maybeChunk,
    std::optional<V0FHEContext> maybeFheContext) {
  auto setMode = [&](concreteprotocol::EncodingInfo::Builder enc) {
    if (!enc.getEncoding().hasIntegerCiphertext()) {
      return;
    }
    auto integerEncodingBuilder = enc.getEncoding().getIntegerCiphertext();

    // Chunks wanted. Setting encoding mode to chunks ...
    if (maybeChunk) {
      integerEncodingBuilder.getMode().setChunked(
          maybeChunk.value().asReader());
      return;
    }

    // Got v0 solution with crt decomposition. Setting encoding mode to crt.
    if (maybeFheContext.has_value()) {
      if (std::holds_alternative<V0Parameter>(maybeFheContext->solution)) {
        auto v0ParameterSol = std::get<V0Parameter>(maybeFheContext->solution);
        if (v0ParameterSol.largeInteger.has_value()) {
          auto moduli = v0ParameterSol.largeInteger->crtDecomposition;
          auto moduliBuilder =
              integerEncodingBuilder.getMode().initCrt().initModuli(
                  moduli.size());
          for (size_t i = 0; i < moduli.size(); i++) {
            moduliBuilder.set(i, moduli[i]);
          }
          return;
        }
      }
    }

    // Got circuit solution with crt decomposition. Setting encoding mode to
    // crt.
    if (maybeFheContext.has_value()) {
      if (std::holds_alternative<optimizer::CircuitSolution>(
              maybeFheContext->solution)) {
        auto circuitSol =
            std::get<optimizer::CircuitSolution>(maybeFheContext->solution);
        if (!circuitSol.crt_decomposition.empty()) {
          auto moduli = circuitSol.crt_decomposition;
          auto moduliBuilder =
              integerEncodingBuilder.getMode().initCrt().initModuli(
                  moduli.size());
          for (size_t i = 0; i < moduli.size(); i++) {
            moduliBuilder.set(i, moduli[i]);
          }
          return;
        }
      }
    }

    // Got nothing particular. Setting encoding mode to native.
    integerEncodingBuilder.getMode().initNative();
  };
  for (auto encInfoBuilder : info.asBuilder().getInputs()) {
    setMode(encInfoBuilder);
  }
  for (auto encInfoBuilder : info.asBuilder().getOutputs()) {
    setMode(encInfoBuilder);
  }
}
} // namespace encodings
} // namespace concretelang
} // namespace mlir
