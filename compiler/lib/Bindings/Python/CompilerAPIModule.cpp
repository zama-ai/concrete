#include "CompilerAPIModule.h"
#include "zamalang-c/Support/CompilerEngine.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOpsDialect.h.inc"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/ExecutionArgument.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

using mlir::zamalang::CompilerEngine;
using mlir::zamalang::ExecutionArgument;

/// Populate the compiler API python module.
void mlir::zamalang::python::populateCompilerAPISubmodule(pybind11::module &m) {
  m.doc() = "Zamalang compiler python API";

  m.def("round_trip", [](std::string mlir_input) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    auto module_ref = mlir::parseSourceString(mlir_input, &context);
    if (!module_ref) {
      throw std::logic_error("mlir parsing failed");
    }
    std::string result;
    llvm::raw_string_ostream os(result);
    module_ref->print(os);
    return os.str();
  });

  pybind11::class_<ExecutionArgument, std::shared_ptr<ExecutionArgument>>(
      m, "ExecutionArgument")
      .def("create",
           pybind11::overload_cast<uint64_t>(&ExecutionArgument::create))
      .def("create", pybind11::overload_cast<std::vector<uint8_t>>(
                         &ExecutionArgument::create))
      .def("is_tensor", &ExecutionArgument::isTensor)
      .def("is_int", &ExecutionArgument::isInt);

  pybind11::class_<CompilerEngine>(m, "CompilerEngine")
      .def(pybind11::init())
      .def("run",
           [](CompilerEngine &engine, std::vector<ExecutionArgument> args) {
             // wrap and call CAPI
             compilerEngine e{&engine};
             exectuionArguments a{args.data(), args.size()};
             return compilerEngineRun(e, a);
           })
      .def("compile_fhe",
           [](CompilerEngine &engine, std::string mlir_input) {
             // wrap and call CAPI
             compilerEngine e{&engine};
             compilerEngineCompile(e, mlir_input.c_str());
           })
      .def("get_compiled_module", &CompilerEngine::getCompiledModule);
}
