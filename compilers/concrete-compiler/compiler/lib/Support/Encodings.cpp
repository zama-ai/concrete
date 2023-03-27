// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/ClientLib/ClientParameters.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Support/Encodings.h>
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Variants.h>
#include <optional>
#include <variant>

namespace FHE = mlir::concretelang::FHE;
namespace clientlib = concretelang::clientlib;

namespace mlir {
namespace concretelang {
namespace encodings {

std::optional<Encoding>
encodingFromType(mlir::Type ty,
                 std::optional<clientlib::ChunkInfo> maybeChunkInfo) {
  if (auto eintTy = ty.dyn_cast<FHE::FheIntegerInterface>()) {
    if (maybeChunkInfo.has_value() &&
        eintTy.getWidth() > maybeChunkInfo.value().size) {
      auto chunkInfo = maybeChunkInfo.value();
      return EncryptedChunkedIntegerScalarEncoding{
          eintTy.getWidth(), eintTy.isSigned(), chunkInfo.width,
          chunkInfo.size};
    } else {
      return EncryptedIntegerScalarEncoding{eintTy.getWidth(),
                                            eintTy.isSigned()};
    }
  } else if (auto eboolTy = ty.dyn_cast<FHE::EncryptedBooleanType>()) {
    return EncryptedBoolScalarEncoding{};
  } else if (auto intTy = ty.dyn_cast<mlir::IntegerType>()) {
    return PlaintextScalarEncoding{intTy.getWidth()};
  } else if (auto indexTy = ty.dyn_cast<mlir::IndexType>()) {
    return IndexScalarEncoding{};
  } else if (auto tensor = ty.dyn_cast<mlir::RankedTensorType>()) {
    std::optional<Encoding> maybeEncoding =
        encodingFromType(tensor.getElementType(), maybeChunkInfo);
    if (maybeEncoding.has_value() &&
        std::holds_alternative<ScalarEncoding>(maybeEncoding.value())) {
      ScalarEncoding scalarEncoding =
          std::get<ScalarEncoding>(maybeEncoding.value());
      return TensorEncoding{scalarEncoding};
    }
  }
  return std::nullopt;
}

llvm::Expected<CircuitEncodings>
getCircuitEncodings(llvm::StringRef functionName, mlir::ModuleOp module,
                    std::optional<clientlib::ChunkInfo> maybeChunkInfo) {

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
  std::vector<Encoding> inputs;
  std::vector<Encoding> outputs;
  for (auto ty : funcType.getInputs()) {
    auto maybeGate = encodingFromType(ty, maybeChunkInfo);
    if (!maybeGate.has_value()) {
      return StreamStringError("Failed to recognize encoding for type : ")
             << ty;
    }
    inputs.push_back(maybeGate.value());
  }
  for (auto ty : funcType.getResults()) {
    auto maybeGate = encodingFromType(ty, maybeChunkInfo);
    if (!maybeGate.has_value()) {
      return StreamStringError("Failed to recognize encoding for type : ")
             << ty;
    }
    outputs.push_back(maybeGate.value());
  }

  return CircuitEncodings{inputs, outputs};
}

bool fromJSON(const llvm::json::Value j, EncryptedIntegerScalarEncoding &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("width", e.width) && O.map("isSigned", e.isSigned);
}
llvm::json::Value toJSON(const EncryptedIntegerScalarEncoding &e) {
  llvm::json::Object object{
      {"width", e.width},
      {"isSigned", e.isSigned},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j,
              EncryptedChunkedIntegerScalarEncoding &e, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("width", e.width) && O.map("isSigned", e.isSigned) &&
         O.map("chunkSize", e.chunkSize) && O.map("chunkWidth", e.chunkWidth);
}
llvm::json::Value toJSON(const EncryptedChunkedIntegerScalarEncoding &e) {
  llvm::json::Object object{
      {"width", e.width},
      {"isSigned", e.isSigned},
      {"chunkSize", e.chunkSize},
      {"chunkWidth", e.chunkWidth},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j, EncryptedBoolScalarEncoding &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O;
}
llvm::json::Value toJSON(const EncryptedBoolScalarEncoding &e) {
  llvm::json::Object object{};
  return object;
}

bool fromJSON(const llvm::json::Value j, PlaintextScalarEncoding &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("width", e.width);
}
llvm::json::Value toJSON(const PlaintextScalarEncoding &e) {
  llvm::json::Object object{{"width", e.width}};
  return object;
}

bool fromJSON(const llvm::json::Value j, IndexScalarEncoding &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O;
}
llvm::json::Value toJSON(const IndexScalarEncoding &e) {
  llvm::json::Object object{};
  return object;
}

bool fromJSON(const llvm::json::Value j, ScalarEncoding &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  if (j.getAsObject()->getObject("EncryptedIntegerScalarEncoding")) {
    return O && O.map("EncryptedIntegerScalarEncoding",
                      std::get<EncryptedIntegerScalarEncoding>(e));
  } else if (j.getAsObject()->getObject(
                 "EncryptedChunkedIntegerScalarEncoding")) {
    return O && O.map("EncryptedChunkedIntegerScalarEncoding",
                      std::get<EncryptedChunkedIntegerScalarEncoding>(e));
  } else if (j.getAsObject()->getObject("EncryptedBoolScalarEncoding")) {
    return O && O.map("EncryptedBoolScalarEncoding",
                      std::get<EncryptedBoolScalarEncoding>(e));
  } else if (j.getAsObject()->getObject("PlaintextScalarEncoding")) {
    return O && O.map("PlaintextScalarEncoding",
                      std::get<PlaintextScalarEncoding>(e));
  } else if (j.getAsObject()->getObject("IndexScalarEncoding")) {
    return O && O.map("IndexScalarEncoding", std::get<IndexScalarEncoding>(e));
  } else {
    return false;
  }
}
llvm::json::Value toJSON(const ScalarEncoding &e) {
  llvm::json::Object object = std::visit(
      overloaded{
          [](EncryptedIntegerScalarEncoding enc) {
            return llvm::json::Object{{"EncryptedIntegerScalarEncoding", enc}};
          },
          [](EncryptedChunkedIntegerScalarEncoding enc) {
            return llvm::json::Object{
                {"EncryptedChunkedIntegerScalarEncoding", enc}};
          },
          [](EncryptedBoolScalarEncoding enc) {
            return llvm::json::Object{{"EncryptedBoolScalarEncoding", enc}};
          },
          [](PlaintextScalarEncoding enc) {
            return llvm::json::Object{{"PlaintextScalarEncoding", enc}};
          },
          [](IndexScalarEncoding enc) {
            return llvm::json::Object{{"IndexScalarEncoding", enc}};
          },
      },
      e);
  return object;
}

bool fromJSON(const llvm::json::Value j, TensorEncoding &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("scalarEncoding", e.scalarEncoding);
}
llvm::json::Value toJSON(const TensorEncoding &e) {
  llvm::json::Object object{{"scalarEncoding", e.scalarEncoding}};
  return object;
}

bool fromJSON(const llvm::json::Value j, Encoding &e, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  if (j.getAsObject()->getObject("ScalarEncoding")) {
    e = EncryptedIntegerScalarEncoding{0, false};
    return O && O.map("ScalarEncoding", std::get<ScalarEncoding>(e));
  } else if (j.getAsObject()->getObject("TensorEncoding")) {
    e = TensorEncoding{EncryptedIntegerScalarEncoding{0, false}};
    return O && O.map("TensorEncoding", std::get<TensorEncoding>(e));
  } else {
    return false;
  }
}
llvm::json::Value toJSON(const Encoding &e) {
  llvm::json::Object object =
      std::visit(overloaded{
                     [](ScalarEncoding enc) {
                       return llvm::json::Object{{"ScalarEncoding", enc}};
                     },
                     [](TensorEncoding enc) {
                       return llvm::json::Object{{"TensorEncoding", enc}};
                     },
                 },
                 e);
  return object;
}

bool fromJSON(const llvm::json::Value j, CircuitEncodings &e,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("inputEncodings", e.inputEncodings) &&
         O.map("outputEncodings", e.outputEncodings);
}
llvm::json::Value toJSON(const CircuitEncodings &e) {
  llvm::json::Object object{{"inputEncodings", e.inputEncodings},
                            {"outputEncodings", e.outputEncodings}};
  return object;
}

} // namespace encodings
} // namespace concretelang
} // namespace mlir
