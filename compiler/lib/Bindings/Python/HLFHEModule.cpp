// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "DialectModules.h"

#include "concretelang-c/Dialect/HLFHE.h"

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

/// Populate the hlfhe python module.
void mlir::concretelang::python::populateDialectHLFHESubmodule(
    pybind11::module &m) {
  m.doc() = "HLFHE dialect Python native extension";

  mlir_type_subclass(m, "EncryptedIntegerType",
                     hlfheTypeIsAnEncryptedIntegerType)
      .def_classmethod("get", [](pybind11::object cls, MlirContext ctx,
                                 unsigned width) {
        // We want the user to receive a python exception for not being able to
        // create the eint
        auto emitException = []() -> mlir::InFlightDiagnostic {
          throw std::invalid_argument("can't create eint with the given width");
        };
        return cls(
            hlfheEncryptedIntegerTypeGetChecked(ctx, width, emitException));
      });
}