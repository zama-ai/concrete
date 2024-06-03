// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Bindings/Python/DialectModules.h"
#include "concretelang/Dialect/FHE/IR/FHEAttrs.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace mlir::concretelang;
using namespace mlir::concretelang::FHE;
using namespace mlir::python::adaptors;

typedef struct {
  MlirType type;
  bool isError;
} MlirTypeOrError;

template <typename T>
MlirTypeOrError IntegerTypeGetChecked(MlirContext ctx, unsigned width) {
  MlirTypeOrError type = {{NULL}, false};
  auto catchError = [&]() -> mlir::InFlightDiagnostic {
    type.isError = true;
    mlir::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
    // The goal here is to make getChecked working, but we don't want the CAPI
    // to stop execution due to an error, and leave the error handling logic to
    // the user of the CAPI
    return engine.emit(mlir::UnknownLoc::get(unwrap(ctx)),
                       mlir::DiagnosticSeverity::Warning);
  };
  T integerType = T::getChecked(catchError, unwrap(ctx), width);
  if (type.isError) {
    return type;
  }
  type.type = wrap(integerType);
  return type;
}

/// Populate the fhe python module.
void mlir::concretelang::python::populateDialectFHESubmodule(
    pybind11::module &m) {
  m.doc() = "FHE dialect Python native extension";

  mlir_type_subclass(m, "EncryptedIntegerType",
                     [](MlirType type) {
                       return unwrap(type).isa<EncryptedUnsignedIntegerType>();
                     })
      .def_classmethod("get", [](pybind11::object cls, MlirContext ctx,
                                 unsigned width) {
        MlirTypeOrError typeOrError =
            IntegerTypeGetChecked<EncryptedUnsignedIntegerType>(ctx, width);
        if (typeOrError.isError) {
          throw std::invalid_argument("can't create eint with the given width");
        }
        return cls(typeOrError.type);
      });

  mlir_type_subclass(m, "EncryptedSignedIntegerType",
                     [](MlirType type) {
                       return unwrap(type).isa<EncryptedSignedIntegerType>();
                     })
      .def_classmethod(
          "get", [](pybind11::object cls, MlirContext ctx, unsigned width) {
            MlirTypeOrError typeOrError =
                IntegerTypeGetChecked<EncryptedSignedIntegerType>(ctx, width);
            if (typeOrError.isError) {
              throw std::invalid_argument(
                  "can't create esint with the given width");
            }
            return cls(typeOrError.type);
          });

  mlir_attribute_subclass(
      m, "PartitionAttr",
      [](MlirAttribute attr) { return unwrap(attr).isa<PartitionAttr>(); })
      .def_classmethod("get", [](pybind11::object cls, MlirContext ctx,
                                 int lweDim, int glweDim, int polySize,
                                 MlirAttribute lweNoiseDistrStdDevAttr,
                                 MlirAttribute glweNoiseDistrStdDevAttr,
                                 int pbsBaseLog, int pbsLevel, int ksBaseLog,
                                 int ksLevel, int msgModulus, int carryModulus,
                                 int maxNoiseLevel, MlirAttribute log2PFailAttr,
                                 bool bigEncryptionKey, int ciphertextModulus) {
        // make sure attributes are of the correct type
        auto lweNoiseDistrStdDev =
            unwrap(lweNoiseDistrStdDevAttr).dyn_cast_or_null<mlir::FloatAttr>();
        if (lweNoiseDistrStdDev == nullptr) {
          throw std::invalid_argument("lweNoiseDistrStdDev isn't a FloatAttr");
        }
        auto glweNoiseDistrStdDev = unwrap(glweNoiseDistrStdDevAttr)
                                        .dyn_cast_or_null<mlir::FloatAttr>();
        if (glweNoiseDistrStdDev == nullptr) {
          throw std::invalid_argument("glweNoiseDistrStdDev isn't a FloatAttr");
        }
        auto log2PFail =
            unwrap(log2PFailAttr).dyn_cast_or_null<mlir::FloatAttr>();
        if (log2PFail == nullptr) {
          throw std::invalid_argument("log2PFail isn't a FloatAttr");
        }

        return cls(wrap(PartitionAttr::get(
            unwrap(ctx), lweDim, glweDim, polySize, lweNoiseDistrStdDev,
            glweNoiseDistrStdDev, pbsBaseLog, pbsLevel, ksBaseLog, ksLevel,
            msgModulus, carryModulus, maxNoiseLevel, log2PFail,
            bigEncryptionKey, ciphertextModulus)));
      });
}
