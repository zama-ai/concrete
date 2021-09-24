#include "DialectModules.h"

#include "zamalang-c/Dialect/HLFHE.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace mlir::zamalang;
using namespace mlir::python::adaptors;

/// Populate the hlfhe python module.
void mlir::zamalang::python::populateDialectHLFHESubmodule(pybind11::module &m) {
  m.doc() = "HLFHE dialect Python native extension";

  mlir_type_subclass(m, "EncryptedIntegerType",
                     hlfheTypeIsAnEncryptedIntegerType)
      .def_classmethod(
          "get", [](pybind11::object cls, MlirContext ctx, unsigned width) {
            return cls(hlfheEncryptedIntegerTypeGet(ctx, width));
          });
}