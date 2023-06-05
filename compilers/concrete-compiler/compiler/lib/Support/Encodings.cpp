// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concrete-protocol.pb.h"
#include "concretelang/Support/Utils.h"
#include "concretelang/Support/V0Parameters.h"
#include <concretelang/ClientLib/ClientParameters.h>
#include <concretelang/Common/Protobuf.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Support/Encodings.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Variants.h>
#include <functional>
#include <memory>
#include <optional>
#include <variant>

namespace FHE = mlir::concretelang::FHE;
namespace clientlib = concretelang::clientlib;

namespace mlir {
namespace concretelang {
namespace encodings {

llvm::Expected<concreteprotocol::EncodingInfo> encodingFromType(mlir::Type ty) {
  auto output = concreteprotocol::EncodingInfo();
  if (auto eintTy = ty.dyn_cast<FHE::FheIntegerInterface>()) {
    auto encoding = new concreteprotocol::IntegerCiphertextEncodingInfo{};
    encoding->set_issigned(eintTy.isSigned());
    encoding->set_width(eintTy.getWidth());
    output.set_allocated_integerciphertext(encoding);
    return output;
  } else if (auto eboolTy = ty.dyn_cast<FHE::EncryptedBooleanType>()) {
    output.set_allocated_booleanciphertext(
        new concreteprotocol::BooleanCiphertextEncodingInfo{});
    return output;
  } else if (auto intTy = ty.dyn_cast<mlir::IntegerType>()) {
    output.set_allocated_plaintext(
        new concreteprotocol::PlaintextEncodingInfo{});
    return output;
  } else if (auto indexTy = ty.dyn_cast<mlir::IndexType>()) {
    output.set_allocated_index(new concreteprotocol::IndexEncodingInfo{});
    return output;
  } else if (auto tensorTy = ty.dyn_cast<mlir::RankedTensorType>()) {
    auto maybeElementEncoding = encodingFromType(tensorTy.getElementType());
    if (!maybeElementEncoding) {
      return maybeElementEncoding.takeError();
    }
    auto output = std::move(*maybeElementEncoding);
    auto shape = new concreteprotocol::Shape{};
    for (auto dim : tensorTy.getShape()) {
      shape->add_dimensions(dim);
    }
    output.set_allocated_shape(shape);
    return output;
  }
  return StreamStringError("Failed to recognize encoding for type : ") << ty;
}

llvm::Expected<concreteprotocol::CircuitEncodingInfo>
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
  auto circuitEncodings = concreteprotocol::CircuitEncodingInfo();
  for (auto ty : funcType.getInputs()) {
    auto maybeEncoding = encodingFromType(ty);
    if (!maybeEncoding) {
      return maybeEncoding.takeError();
    }
    circuitEncodings.mutable_inputs()->AddAllocated(
        new concreteprotocol::EncodingInfo(*maybeEncoding));
  }
  for (auto ty : funcType.getResults()) {
    auto maybeEncoding = encodingFromType(ty);
    if (!maybeEncoding) {
      return maybeEncoding.takeError();
    }
    circuitEncodings.mutable_outputs()->AddAllocated(
        new concreteprotocol::EncodingInfo(*maybeEncoding));
  }

  return circuitEncodings;
}

void setCircuitEncodingModes(
    concreteprotocol::CircuitEncodingInfo &info,
    std::optional<concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode>
        maybeChunk,
    std::optional<V0FHEContext> maybeFheContext) {
  auto setMode = [&](concreteprotocol::EncodingInfo *enc) {
    if (enc->has_integerciphertext()) {
      auto intEnc = enc->mutable_integerciphertext();
      if (maybeChunk) {
        intEnc->set_allocated_chunked(
            new concreteprotocol::IntegerCiphertextEncodingInfo::ChunkedMode(
                maybeChunk.value()));
        return;
      }
      if (maybeFheContext.has_value()) {
        if (std::holds_alternative<V0Parameter>(maybeFheContext->solution)) {
          auto v0ParameterSol =
              std::get<V0Parameter>(maybeFheContext->solution);
          if (v0ParameterSol.largeInteger.has_value()) {
            auto moduli = v0ParameterSol.largeInteger->crtDecomposition;
            auto crt =
                new concreteprotocol::IntegerCiphertextEncodingInfo::CrtMode();
            crt->mutable_moduli()->Assign(moduli.begin(), moduli.end());
            intEnc->set_allocated_crt(crt);
            return;
          }
        } else if (std::holds_alternative<optimizer::CircuitSolution>(
                       maybeFheContext->solution)) {
          auto circuitSol =
              std::get<optimizer::CircuitSolution>(maybeFheContext->solution);
          if (!circuitSol.crt_decomposition.empty()) {
            auto moduli = circuitSol.crt_decomposition;
            auto crt =
                new concreteprotocol::IntegerCiphertextEncodingInfo::CrtMode();
            crt->mutable_moduli()->Assign(moduli.begin(), moduli.end());
            intEnc->set_allocated_crt(crt);
            return;
          }
        }
      }
      auto native =
          new concreteprotocol::IntegerCiphertextEncodingInfo::NativeMode();
      intEnc->set_allocated_native(native);
    }
  };
  for (int i = 0; i < info.inputs_size(); i++) {
    auto enc = info.mutable_inputs(i);
    setMode(enc);
  }
  for (int i = 0; i < info.outputs_size(); i++) {
    auto enc = info.mutable_outputs(i);
    setMode(enc);
  }
}
} // namespace encodings
} // namespace concretelang
} // namespace mlir
