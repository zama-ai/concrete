// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concrete-protocol.pb.h"
#include <concretelang/ClientLib/ClientParameters.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Support/Encodings.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Variants.h>
#include <memory>
#include <optional>
#include <variant>

namespace FHE = mlir::concretelang::FHE;
namespace clientlib = concretelang::clientlib;
namespace protocol = concreteprotocol;

namespace mlir {
namespace concretelang {
namespace encodings {

llvm::Expected<std::unique_ptr<protocol::EncodingInfo>>
encodingFromType(mlir::Type ty) {
  auto output = std::make_unique<protocol::EncodingInfo>();
  if (auto eintTy = ty.dyn_cast<FHE::FheIntegerInterface>()) {
    auto encoding = new protocol::IntegerCiphertextEncodingInfo{};
    encoding->set_issigned(eintTy.isSigned());
    encoding->set_width(eintTy.getWidth());
    output->set_allocated_integerciphertext(encoding);
    return output;
  } else if (auto eboolTy = ty.dyn_cast<FHE::EncryptedBooleanType>()) {
    output->set_allocated_booleanciphertext(
        new protocol::BooleanCiphertextEncodingInfo{});
    return output;
  } else if (auto intTy = ty.dyn_cast<mlir::IntegerType>()) {
    output->set_allocated_plaintext(new protocol::PlaintextEncodingInfo{});
    return output;
  } else if (auto indexTy = ty.dyn_cast<mlir::IndexType>()) {
    output->set_allocated_index(new protocol::IndexEncodingInfo{});
    return output;
  } else if (auto tensorTy = ty.dyn_cast<mlir::RankedTensorType>()) {
    auto maybeElementEncoding = encodingFromType(tensorTy.getElementType());
    if (!maybeElementEncoding) {
      return maybeElementEncoding.takeError();
    }
    auto output = std::move(*maybeElementEncoding);
    auto shape = new protocol::Shape{};
    for (auto dim : tensorTy.getShape()) {
      shape->add_dimensions(dim);
    }
    output->set_allocated_shape(shape);
    return output;
  }
  return StreamStringError("Failed to recognize encoding for type : ") << ty;
}

llvm::Expected<std::unique_ptr<protocol::CircuitEncodingInfo>>
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
  auto circuitEncodings = std::make_unique<protocol::CircuitEncodingInfo>();
  for (auto ty : funcType.getInputs()) {
    auto maybeEncoding = encodingFromType(ty);
    if (!maybeEncoding) {
      return maybeEncoding.takeError();
    }
    circuitEncodings->mutable_inputs()->AddAllocated((*maybeEncoding).release());
  }
  for (auto ty : funcType.getResults()) {
    auto maybeEncoding = encodingFromType(ty);
    if (!maybeEncoding) {
      return maybeEncoding.takeError();
    }
    circuitEncodings->mutable_outputs()->AddAllocated((*maybeEncoding).release());
  }

  return circuitEncodings;
}

} // namespace encodings
} // namespace concretelang
} // namespace mlir
