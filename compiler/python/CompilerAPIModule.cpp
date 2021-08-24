#include "CompilerAPIModule.h"
#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/CompilerTools.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

using namespace zamalang;
using mlir::zamalang::CompilerEngine;

/// Populate the compiler API python module.
void zamalang::python::populateCompilerAPISubmodule(pybind11::module &m) {
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

  pybind11::class_<CompilerEngine>(m, "CompilerEngine")
      .def(pybind11::init())
      .def("run",
           [](CompilerEngine &engine, std::vector<uint64_t> args) {
             auto result = engine.run(args);
             if (!result) {
               llvm::errs()
                   << "Execution failed: " << result.takeError() << "\n";
               throw std::runtime_error(
                   "failed running, see previous logs for more info");
             }
             return result.get();
           })
      .def("compile_fhe",
           [](CompilerEngine &engine, std::string mlir_input) {
             auto result = engine.compile(mlir_input);
             if (!result) {
               llvm::errs() << "Compilation failed: " << result << "\n";
               throw std::runtime_error(
                   "failed compiling, see previous logs for more info");
             }
           })
      .def("get_compiled_module", &CompilerEngine::getCompiledModule);
}
