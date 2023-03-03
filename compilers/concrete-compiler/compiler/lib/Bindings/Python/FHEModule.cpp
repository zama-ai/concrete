// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang-c/Dialect/FHE.h"
#include "concretelang/Bindings/Python/DialectModules.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace mlir::concretelang;
using namespace mlir::python::adaptors;

/// Populate the fhe python module.
void mlir::concretelang::python::populateDialectFHESubmodule(
    pybind11::module &m) {
  m.doc() = "FHE dialect Python native extension";

  mlir_type_subclass(m, "EncryptedIntegerType", fheTypeIsAnEncryptedIntegerType)
      .def_classmethod("get", [](pybind11::object cls, MlirContext ctx,
                                 unsigned width) {
        MlirTypeOrError typeOrError =
            fheEncryptedIntegerTypeGetChecked(ctx, width);
        if (typeOrError.isError) {
          throw std::invalid_argument("can't create eint with the given width");
        }
        return cls(typeOrError.type);
      });

  mlir_type_subclass(m, "EncryptedSignedIntegerType",
                     fheTypeIsAnEncryptedSignedIntegerType)
      .def_classmethod(
          "get", [](pybind11::object cls, MlirContext ctx, unsigned width) {
            MlirTypeOrError typeOrError =
                fheEncryptedSignedIntegerTypeGetChecked(ctx, width);
            if (typeOrError.isError) {
              throw std::invalid_argument(
                  "can't create esint with the given width");
            }
            return cls(typeOrError.type);
          });
}
