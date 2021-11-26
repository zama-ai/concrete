#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Support/ClientParameters.h"
#include "zamalang/Support/V0Curves.h"

namespace mlir {
namespace zamalang {

const auto securityLevel = SECURITY_LEVEL_128;
const auto keyFormat = KEY_FORMAT_BINARY;
const auto v0Curve = getV0Curves(securityLevel, keyFormat);

// For the v0 the secretKeyID and precision are the same for all gates.
llvm::Expected<CircuitGate> gateFromMLIRType(std::string secretKeyID,
                                             Precision precision,
                                             Variance variance,
                                             mlir::Type type) {
  if (type.isIntOrIndex()) {
    // TODO - The index type is dependant of the target architecture, so
    // actually we assume we target only 64 bits, we need to have some the size
    // of the word of the target system.
    size_t width = 64;
    if (!type.isIndex()) {
      width = type.getIntOrFloatBitWidth();
    }
    return CircuitGate{
        /*.encryption = */ llvm::None,
        /*.shape = */
        {
            /*.width = */ width,
            /*.dimensions = */ std::vector<int64_t>(),
            /*.size = */ 0,
        },
    };
  }
  if (type.isa<mlir::zamalang::LowLFHE::LweCiphertextType>()) {
    // TODO - Get the width from the LWECiphertextType instead of global
    // precision (could be possible after merge lowlfhe-ciphertext-parameter)
    return CircuitGate{
        .encryption = llvm::Optional<EncryptionGate>({
            .secretKeyID = secretKeyID,
            .variance = variance,
            .encoding = {.precision = precision},
        }),
        /*.shape = */
        {
            /*.width = */ precision,
            /*.dimensions = */ std::vector<int64_t>(),
            /*.size = */ 0,
        },
    };
  }
  auto tensor = type.dyn_cast_or_null<mlir::RankedTensorType>();
  if (tensor != nullptr) {
    auto gate = gateFromMLIRType(secretKeyID, precision, variance,
                                 tensor.getElementType());
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    gate->shape.dimensions = tensor.getShape().vec();
    gate->shape.size = 1;
    for (auto dimSize : gate->shape.dimensions) {
      gate->shape.size *= dimSize;
    }
    return gate;
  }
  return llvm::make_error<llvm::StringError>(
      "cannot convert MLIR type to shape", llvm::inconvertibleErrorCode());
}

llvm::Expected<ClientParameters>
createClientParametersForV0(V0FHEContext fheContext, llvm::StringRef name,
                            mlir::ModuleOp module) {
  auto v0Param = fheContext.parameter;
  Variance encryptionVariance =
      v0Curve->getVariance(1, 1 << v0Param.polynomialSize, 64);
  Variance keyswitchVariance = v0Curve->getVariance(1, v0Param.nSmall, 64);
  // Static client parameters from global parameters for v0
  ClientParameters c = {};
  c.secretKeys = {
      {"small", {/*.size = */ v0Param.nSmall}},
      {"big", {/*.size = */ v0Param.getNBigGlweSize()}},
  };
  c.bootstrapKeys = {
      {
          "bsk_v0",
          {
              /*.inputSecretKeyID = */ "small",
              /*.outputSecretKeyID = */ "big",
              /*.level = */ v0Param.brLevel,
              /*.baseLog = */ v0Param.brLogBase,
              /*.k = */ v0Param.k,
              /*.variance = */ encryptionVariance,
          },
      },
  };
  c.keyswitchKeys = {
      {
          "ksk_v0",
          {
              /*.inputSecretKeyID = */ "big",
              /*.outputSecretKeyID = */ "small",
              /*.level = */ v0Param.ksLevel,
              /*.baseLog = */ v0Param.ksLogBase,
              /*.variance = */ keyswitchVariance,
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
  auto precision = fheContext.constraint.p;

  // Create input and output circuit gate parameters
  auto funcType = (*funcOp).getType();
  bool hasContext =
      funcType.getInputs().back().isa<mlir::zamalang::LowLFHE::ContextType>();
  for (auto inType = funcType.getInputs().begin();
       inType < funcType.getInputs().end() - hasContext; inType++) {
    auto gate = gateFromMLIRType("big", precision, encryptionVariance, *inType);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    c.inputs.push_back(gate.get());
  }
  for (auto outType : funcType.getResults()) {
    auto gate = gateFromMLIRType("big", precision, encryptionVariance, outType);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    c.outputs.push_back(gate.get());
  }
  return c;
}

} // namespace zamalang
} // namespace mlir
