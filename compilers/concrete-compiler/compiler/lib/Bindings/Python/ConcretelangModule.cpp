// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Bindings/Python/CompilerAPIModule.h"
#include "concretelang/Bindings/Python/DialectModules.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"
#include "concretelang/Support/Constants.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace mlir::concretelang::FHE;
using namespace mlir::concretelang::FHELinalg;
using namespace mlir::concretelang::Tracing;

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FHE, fhe);
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FHE, fhe, FHEDialect)

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FHELinalg, fhelinalg);
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FHELinalg, fhelinalg, FHELinalgDialect)

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TRACING, tracing);
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tracing, tracing, TracingDialect)

PYBIND11_MODULE(_concretelang, m) {
  m.doc() = "Concretelang Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  m.def(
      "register_dialects",
      [](py::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        const MlirContext context =
            mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        const MlirDialectRegistry registry = mlirDialectRegistryCreate();
        mlirRegisterAllDialects(registry);
        mlirContextAppendDialectRegistry(context, registry);

        const MlirDialectHandle fhe = mlirGetDialectHandle__fhe__();
        mlirDialectHandleRegisterDialect(fhe, context);

        const MlirDialectHandle fhelinalg = mlirGetDialectHandle__fhelinalg__();
        mlirDialectHandleRegisterDialect(fhelinalg, context);

        const MlirDialectHandle tracing = mlirGetDialectHandle__tracing__();
        mlirDialectHandleRegisterDialect(tracing, context);

        mlirContextLoadAllAvailableDialects(context);
      },
      "Register Concretelang dialects on a PyMlirContext.");

  py::module fhe = m.def_submodule("_fhe", "FHE API");
  mlir::concretelang::python::populateDialectFHESubmodule(fhe);

  py::module api = m.def_submodule("_compiler", "Compiler API");
  mlir::concretelang::python::populateCompilerAPISubmodule(api);
}
