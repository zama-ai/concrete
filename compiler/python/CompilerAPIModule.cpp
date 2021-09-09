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
using zamalang::python::ExecutionArgument;

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
      .def(
          "run",
          [](CompilerEngine &engine, std::vector<ExecutionArgument> args) {
            auto maybeArgument = engine.buildArgument();
            if (auto err = maybeArgument.takeError()) {
              llvm::errs() << "Execution failed: " << err << "\n";
              throw std::runtime_error(
                  "failed building arguments, see previous logs for more info");
            }
            // Set the integer/tensor arguments
            auto arguments = std::move(maybeArgument.get());
            for (auto i = 0; i < args.size(); i++) {
              if (args[i].isInt()) { // integer argument
                if (auto err =
                        arguments->setArg(i, args[i].getIntegerArgument())) {
                  llvm::errs() << "Execution failed: " << err << "\n";
                  throw std::runtime_error(
                      "failed pushing integer argument, see "
                      "previous logs for more info");
                }
              } else { // tensor argument
                assert(args[i].isTensor() && "should be tensor argument");
                if (auto err = arguments->setArg(i, args[i].getTensorArgument(),
                                                 args[i].getTensorSize())) {
                  llvm::errs() << "Execution failed: " << err << "\n";
                  throw std::runtime_error(
                      "failed pushing tensor argument, see "
                      "previous logs for more info");
                }
              }
            }
            // Invoke the lambda
            if (auto err = engine.invoke(*arguments)) {
              llvm::errs() << "Execution failed: " << err << "\n";
              throw std::runtime_error(
                  "failed running, see previous logs for more info");
            }
            uint64_t result = 0;
            if (auto err = arguments->getResult(0, result)) {
              llvm::errs() << "Execution failed: " << err << "\n";
              throw std::runtime_error(
                  "failed getting result, see previous logs for more info");
            }
            return result;
          })
      .def("compile_fhe",
           [](CompilerEngine &engine, std::string mlir_input) {
             auto error = engine.compile(mlir_input);
             if (error) {
               llvm::errs() << "Compilation failed: " << error << "\n";
               throw std::runtime_error(
                   "failed compiling, see previous logs for more info");
             }
           })
      .def("get_compiled_module", &CompilerEngine::getCompiledModule);
}
