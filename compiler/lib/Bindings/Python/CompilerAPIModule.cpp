#include "CompilerAPIModule.h"
#include "zamalang-c/Support/CompilerEngine.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOpsDialect.h.inc"
#include "zamalang/Support/ExecutionArgument.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/JitCompilerEngine.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

using mlir::zamalang::ExecutionArgument;
using mlir::zamalang::JitCompilerEngine;
using mlir::zamalang::LambdaArgument;

/// Populate the compiler API python module.
void mlir::zamalang::python::populateCompilerAPISubmodule(pybind11::module &m) {
  m.doc() = "Zamalang compiler python API";

  m.def("round_trip",
        [](std::string mlir_input) { return roundTrip(mlir_input.c_str()); });

  pybind11::class_<ExecutionArgument, std::shared_ptr<ExecutionArgument>>(
      m, "ExecutionArgument")
      .def("create",
           pybind11::overload_cast<uint64_t>(&ExecutionArgument::create))
      .def("create", pybind11::overload_cast<std::vector<uint8_t>>(
                         &ExecutionArgument::create))
      .def("is_tensor", &ExecutionArgument::isTensor)
      .def("is_int", &ExecutionArgument::isInt);

  pybind11::class_<JitCompilerEngine>(m, "JitCompilerEngine")
      .def(pybind11::init())
      .def_static("build_lambda",
                  [](std::string mlir_input, std::string func_name) {
                    return buildLambda(mlir_input.c_str(), func_name.c_str());
                  });

  pybind11::class_<lambdaArgument>(m, "LambdaArgument")
      .def("is_tensor",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsTensor(lambda_arg);
           })
      .def("get_tensor_data",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetTensorData(lambda_arg);
           })
      .def("get_tensor_shape",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetTensorDimensions(lambda_arg);
           })
      .def("is_scalar",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsScalar(lambda_arg);
           })
      .def("get_scalar", [](lambdaArgument &lambda_arg) {
        return lambdaArgumentGetScalar(lambda_arg);
      });

  pybind11::class_<JitCompilerEngine::Lambda>(m, "Lambda")
      .def("invoke", [](JitCompilerEngine::Lambda &py_lambda,
                        std::vector<ExecutionArgument> args) {
        // wrap and call CAPI
        lambda c_lambda{&py_lambda};
        exectuionArguments a{args.data(), args.size()};
        return invokeLambda(c_lambda, a);
      });
}
