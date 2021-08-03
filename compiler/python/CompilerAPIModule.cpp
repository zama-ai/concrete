#include "CompilerAPIModule.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include <mlir/Parser.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>

using namespace zamalang;

/// Populate the compiler API python module.
void zamalang::python::populateCompilerAPISubmodule(pybind11::module &m) {
  m.doc() = "Zamalang compiler python API";

  m.def("round_trip", [](std::string mlir_input) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    auto mlir_module = mlir::parseSourceString(mlir_input, &context);
    if (!mlir_module) {
      throw std::logic_error("mlir parsing failed");
    }
    std::string result;
    llvm::raw_string_ostream os(result);
    mlir_module->print(os);
    return os.str();
  });
}