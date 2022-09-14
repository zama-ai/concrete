// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.
#include <map>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "concrete/curves.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Conversion/Utils/GlobalFHEContext.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Support/Error.h"

namespace mlir {
namespace concretelang {

namespace clientlib = ::concretelang::clientlib;
using ::concretelang::clientlib::CircuitGate;
using ::concretelang::clientlib::ClientParameters;
using ::concretelang::clientlib::Encoding;
using ::concretelang::clientlib::EncryptionGate;
using ::concretelang::clientlib::LweSecretKeyID;
using ::concretelang::clientlib::Precision;
using ::concretelang::clientlib::Variance;

const auto keyFormat = concrete::BINARY;

/// For the v0 the secretKeyID and precision are the same for all gates.
llvm::Expected<CircuitGate> gateFromMLIRType(V0FHEContext fheContext,
                                             LweSecretKeyID secretKeyID,
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

    bool sign = type.isSignedInteger();

    return CircuitGate{
        /*.encryption = */ llvm::None,
        /*.shape = */
        {/*.width = */ width,
         /*.dimensions = */ std::vector<int64_t>(),
         /*.size = */ 0,
         /* .sign */ sign},
    };
  }
  if (auto lweTy = type.dyn_cast_or_null<
                   mlir::concretelang::FHE::FheIntegerInterface>()) {
    bool sign = lweTy.isSigned();
    std::vector<int64_t> crt;
    if (fheContext.parameter.largeInteger.has_value()) {
      crt = fheContext.parameter.largeInteger.value().crtDecomposition;
    }
    return CircuitGate{
        /* .encryption = */ llvm::Optional<EncryptionGate>({
            /* .secretKeyID = */ secretKeyID,
            /* .variance = */ variance,
            /* .encoding = */
            {
                /* .precision = */ lweTy.getWidth(),
                /* .crt = */ crt,
                /*.sign = */ sign,
            },
        }),
        /*.shape = */
        {/*.width = */ (size_t)lweTy.getWidth(),
         /*.dimensions = */ std::vector<int64_t>(),
         /*.size = */ 0,
         /*.sign = */ sign},
    };
  }
  if (auto lweTy = type.dyn_cast_or_null<
                   mlir::concretelang::FHE::EncryptedBooleanType>()) {
    size_t width = mlir::concretelang::FHE::EncryptedBooleanType::getWidth();
    return CircuitGate{
        /* .encryption = */ llvm::Optional<EncryptionGate>({
            /* .secretKeyID = */ secretKeyID,
            /* .variance = */ variance,
            /* .encoding = */
            {
                /* .precision = */ width,
                /* .crt = */ std::vector<int64_t>(),
            },
        }),
        /*.shape = */
        {
            /*.width = */ width,
            /*.dimensions = */ std::vector<int64_t>(),
            /*.size = */ 0,
            /*.sign = */ false,
        },
    };
  }
  auto tensor = type.dyn_cast_or_null<mlir::RankedTensorType>();
  if (tensor != nullptr) {
    auto gate = gateFromMLIRType(fheContext, secretKeyID, variance,
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
createClientParametersForV0(V0FHEContext fheContext,
                            llvm::StringRef functionName, mlir::ModuleOp module,
                            int bitsOfSecurity) {
  const auto v0Curve = concrete::getSecurityCurve(bitsOfSecurity, keyFormat);

  if (v0Curve == nullptr) {
    return StreamStringError("Cannot find security curves for ")
           << bitsOfSecurity << "bits";
  }

  V0Parameter &v0Param = fheContext.parameter;
  Variance inputVariance =
      v0Curve->getVariance(1, v0Param.getNBigLweDimension(), 64);

  Variance bootstrapKeyVariance = v0Curve->getVariance(
      v0Param.glweDimension, v0Param.getPolynomialSize(), 64);
  Variance keyswitchKeyVariance = v0Curve->getVariance(1, v0Param.nSmall, 64);
  // Static client parameters from global parameters for v0
  ClientParameters c;
  c.secretKeys = {
      {clientlib::BIG_KEY, {/*.size = */ v0Param.getNBigLweDimension()}},
  };
  bool has_small_key = v0Param.nSmall != 0;
  bool has_bootstrap = v0Param.brLevel != 0;
  if (has_small_key) {
    c.secretKeys.insert({clientlib::SMALL_KEY, {/*.size = */ v0Param.nSmall}});
  }
  if (has_bootstrap) {
    auto inputKey = (has_small_key) ? clientlib::SMALL_KEY : clientlib::BIG_KEY;
    c.bootstrapKeys = {
        {
            clientlib::BOOTSTRAP_KEY,
            {
                /*.inputSecretKeyID = */ inputKey,
                /*.outputSecretKeyID = */ clientlib::BIG_KEY,
                /*.level = */ v0Param.brLevel,
                /*.baseLog = */ v0Param.brLogBase,
                /*.glweDimension = */ v0Param.glweDimension,
                /*.variance = */ bootstrapKeyVariance,
            },
        },
    };
  }
  if (v0Param.largeInteger.hasValue()) {
    clientlib::PackingKeySwitchParam param;
    param.inputSecretKeyID = clientlib::BIG_KEY;
    param.outputSecretKeyID = clientlib::BIG_KEY;
    param.level = v0Param.largeInteger->wopPBS.packingKeySwitch.level;
    param.baseLog = v0Param.largeInteger->wopPBS.packingKeySwitch.baseLog;
    param.bootstrapKeyID = clientlib::BOOTSTRAP_KEY;
    param.variance = v0Curve->getVariance(v0Param.glweDimension,
                                          v0Param.getPolynomialSize(), 64);
    c.packingKeys = {
        {
            "fpksk_v0",
            param,
        },
    };
  }
  if (has_small_key) {
    c.keyswitchKeys = {
        {
            clientlib::KEYSWITCH_KEY,
            {
                /*.inputSecretKeyID = */ clientlib::BIG_KEY,
                /*.outputSecretKeyID = */ clientlib::SMALL_KEY,
                /*.level = */ v0Param.ksLevel,
                /*.baseLog = */ v0Param.ksLogBase,
                /*.variance = */ keyswitchKeyVariance,
            },
        },
    };
  }

  c.functionName = (std::string)functionName;
  // Find the input function
  auto rangeOps = module.getOps<mlir::func::FuncOp>();
  auto funcOp = llvm::find_if(rangeOps, [&](mlir::func::FuncOp op) {
    return op.getName() == functionName;
  });
  if (funcOp == rangeOps.end()) {
    return StreamStringError(
               "cannot find the function for generate client parameters: ")
           << functionName;
  }

  // Create input and output circuit gate parameters
  auto funcType = (*funcOp).getFunctionType();

  auto inputs = funcType.getInputs();

  auto gateFromType = [&](mlir::Type ty) {
    return gateFromMLIRType(fheContext, clientlib::BIG_KEY, inputVariance, ty);
  };
  for (auto inType : inputs) {
    auto gate = gateFromType(inType);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    c.inputs.push_back(gate.get());
  }
  for (auto outType : funcType.getResults()) {
    auto gate = gateFromType(outType);
    if (auto err = gate.takeError()) {
      return std::move(err);
    }
    c.outputs.push_back(gate.get());
  }
  return c;
}

} // namespace concretelang
} // namespace mlir
