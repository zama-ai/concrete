// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "CompilerAPIModule.h"
#include "DialectModules.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "concretelang-c/Dialect/HLFHE.h"
#include "concretelang-c/Dialect/HLFHELinalg.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_concretelang, m) {
  m.doc() = "Concretelang Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  m.def(
      "register_dialects",
      [](py::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        // Collect Concretelang dialects to register.
        MlirDialectHandle hlfhe = mlirGetDialectHandle__hlfhe__();
        mlirDialectHandleRegisterDialect(hlfhe, context);
        mlirDialectHandleLoadDialect(hlfhe, context);
        MlirDialectHandle hlfhelinalg = mlirGetDialectHandle__hlfhelinalg__();
        mlirDialectHandleRegisterDialect(hlfhelinalg, context);
        mlirDialectHandleLoadDialect(hlfhelinalg, context);
      },
      "Register Concretelang dialects on a PyMlirContext.");

  py::module hlfhe = m.def_submodule("_hlfhe", "HLFHE API");
  mlir::concretelang::python::populateDialectHLFHESubmodule(hlfhe);

  py::module api = m.def_submodule("_compiler", "Compiler API");
  mlir::concretelang::python::populateCompilerAPISubmodule(api);
}