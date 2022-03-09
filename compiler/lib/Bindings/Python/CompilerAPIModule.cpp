// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "CompilerAPIModule.h"
#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/Dialect/FHE/IR/FHEOpsDialect.h.inc"
#include "concretelang/Support/Jit.h"
#include "concretelang/Support/JitCompilerEngine.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

using mlir::concretelang::JitCompilerEngine;
using mlir::concretelang::LambdaArgument;

const char *noEmptyStringPtr(std::string &s) {
  return (s.empty()) ? nullptr : s.c_str();
}

/// Populate the compiler API python module.
void mlir::concretelang::python::populateCompilerAPISubmodule(
    pybind11::module &m) {
  m.doc() = "Concretelang compiler python API";

  m.def("round_trip",
        [](std::string mlir_input) { return roundTrip(mlir_input.c_str()); });

  m.def("library",
        [](std::string library_path, std::vector<std::string> mlir_modules) {
          return library(library_path, mlir_modules);
        });

  pybind11::class_<JitCompilerEngine>(m, "JitCompilerEngine")
      .def(pybind11::init())
      .def_static("build_lambda",
                  [](std::string mlir_input, std::string func_name,
                     std::string runtime_lib_path, std::string keysetcache_path,
                     bool auto_parallelize, bool loop_parallelize,
                     bool df_parallelize) {
                    return buildLambda(mlir_input.c_str(), func_name.c_str(),
                                       noEmptyStringPtr(runtime_lib_path),
                                       noEmptyStringPtr(keysetcache_path),
                                       auto_parallelize, loop_parallelize,
                                       df_parallelize);
                  });

  pybind11::class_<lambdaArgument>(m, "LambdaArgument")
      .def_static("from_tensor",
                  [](std::vector<uint8_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU8(tensor, dims);
                  })
      .def_static("from_tensor",
                  [](std::vector<uint16_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU16(tensor, dims);
                  })
      .def_static("from_tensor",
                  [](std::vector<uint32_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU32(tensor, dims);
                  })
      .def_static("from_tensor",
                  [](std::vector<uint64_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU64(tensor, dims);
                  })
      .def_static("from_scalar", lambdaArgumentFromScalar)
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
                        std::vector<lambdaArgument> args) {
        // wrap and call CAPI
        lambda c_lambda{&py_lambda};
        executionArguments a{args.data(), args.size()};
        return invokeLambda(c_lambda, a);
      });
}
