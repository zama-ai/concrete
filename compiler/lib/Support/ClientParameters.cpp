#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Support/ClientParameters.h"

namespace mlir {
namespace zamalang {

// For the v0 the secretKeyID and precision are the same for all gates.
llvm::Expected<CircuitGate> gateFromMLIRType(std::string secretKeyID,
                                             Precision precision,
                                             mlir::Type type) {
  if (type.isInteger(64)) {
    return CircuitGate{
        .encryption = llvm::None,
        .shape = {.size = 0},
    };
  }
  if (type.isa<mlir::zamalang::LowLFHE::LweCiphertextType>()) {
    return CircuitGate{
        .encryption = llvm::Optional<EncryptionGate>({
            .secretKeyID = secretKeyID,
            // TODO - Compute variance, wait for security estimator
            .variance = 0.,
            .encoding = {.precision = precision},
        }),
        .shape = {.size = 0},
    };
  }
  auto memref = type.dyn_cast_or_null<mlir::MemRefType>();
  if (memref != nullptr) {
    auto gate =
        gateFromMLIRType(secretKeyID, precision, memref.getElementType());
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    gate->shape.size = memref.getDimSize(0);
    return gate;
  }
  return llvm::make_error<llvm::StringError>(
      "cannot convert MLIR type to shape", llvm::inconvertibleErrorCode());
}

llvm::Expected<ClientParameters>
createClientParametersForV0(V0Parameter &v0Param, Precision precision,
                            llvm::StringRef name, mlir::ModuleOp module) {
  // Static client parameters from global parameters for v0
  ClientParameters c{
      .secretKeys{
          {"small", {.size = v0Param.nSmall}},
          {"big", {.size = v0Param.k * (1 << v0Param.polynomialSize)}},
      },
      .bootstrapKeys{
          {
              "bsk_v0",
              {
                  .inputSecretKeyID = "small",
                  .outputSecretKeyID = "big",
                  .level = v0Param.brLevel,
                  .baseLog = v0Param.brLogBase,
                  .k = v0Param.k,
                  // TODO - Compute variance, wait for security estimator
                  .variance = 0,
              },
          },
      },
      .keyswitchKeys{
          {
              "ksk_v0",
              {
                  .inputSecretKeyID = "big",
                  .outputSecretKeyID = "small",
                  .level = v0Param.ksLevel,
                  .baseLog = v0Param.ksLogBase,
                  // TODO - Compute variance, wait for security estimator
                  .variance = 0,
              },
          },
      },
  };
  // Find the input function
  auto rangeOps = module.getOps<mlir::FuncOp>();
  auto funcOp = llvm::find_if(
      rangeOps, [&](mlir::FuncOp op) { return op.getName() == name; });
  if (funcOp == rangeOps.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find the function for generate client parameters",
        llvm::inconvertibleErrorCode());
  }

  // For the v0 the precision is global
  Encoding encoding = {.precision = precision};

  // Create input and output circuit gate parameters
  auto funcType = (*funcOp).getType();
  for (auto inType : funcType.getInputs()) {
    auto gate = gateFromMLIRType("big", precision, inType);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    c.inputs.push_back(gate.get());
  }
  for (auto outType : funcType.getResults()) {
    auto gate = gateFromMLIRType("big", precision, outType);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    c.outputs.push_back(gate.get());
  }
  return c;
}

} // namespace zamalang
} // namespace mlir