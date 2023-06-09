// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Bindings/Python/CompilerAPIModule.h"
#include "concretelang/Bindings/Python/CompilerEngine.h"
#include "concretelang/Dialect/FHE/IR/FHEOpsDialect.h.inc"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/Jit.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/OptUtils.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

using mlir::concretelang::CompilationOptions;
using mlir::concretelang::JITSupport;
using mlir::concretelang::LambdaArgument;

/// Populate the compiler API python module.
void mlir::concretelang::python::populateCompilerAPISubmodule(
    pybind11::module &m) {
  m.doc() = "Concretelang compiler python API";

  m.def("round_trip",
        [](std::string mlir_input) { return roundTrip(mlir_input.c_str()); });

  m.def("terminate_df_parallelization", &terminateDataflowParallelization);

  m.def("init_df_parallelization", &initDataflowParallelization);

  pybind11::enum_<optimizer::Strategy>(m, "OptimizerStrategy")
      .value("V0", optimizer::Strategy::V0)
      .value("DAG_MONO", optimizer::Strategy::DAG_MONO)
      .value("DAG_MULTI", optimizer::Strategy::DAG_MULTI)
      .export_values();

  pybind11::class_<CompilationOptions>(m, "CompilationOptions")
      .def(pybind11::init(
          [](std::string funcname) { return CompilationOptions(funcname); }))
      .def("set_funcname",
           [](CompilationOptions &options, std::string funcname) {
             options.clientParametersFuncName = funcname;
           })
      .def("set_verify_diagnostics",
           [](CompilationOptions &options, bool b) {
             options.verifyDiagnostics = b;
           })
      .def("set_auto_parallelize", [](CompilationOptions &options,
                                      bool b) { options.autoParallelize = b; })
      .def("set_loop_parallelize", [](CompilationOptions &options,
                                      bool b) { options.loopParallelize = b; })
      .def("set_dataflow_parallelize",
           [](CompilationOptions &options, bool b) {
             options.dataflowParallelize = b;
           })
      .def("set_optimize_concrete", [](CompilationOptions &options,
                                       bool b) { options.optimizeTFHE = b; })
      .def("set_p_error",
           [](CompilationOptions &options, double p_error) {
             options.optimizerConfig.p_error = p_error;
           })
      .def("set_display_optimizer_choice",
           [](CompilationOptions &options, bool display) {
             options.optimizerConfig.display = display;
           })
      .def("set_optimizer_strategy",
           [](CompilationOptions &options, optimizer::Strategy strategy) {
             options.optimizerConfig.strategy = strategy;
           })
      .def("set_global_p_error",
           [](CompilationOptions &options, double global_p_error) {
             options.optimizerConfig.global_p_error = global_p_error;
           })
      .def("set_security_level",
           [](CompilationOptions &options, int security_level) {
             options.optimizerConfig.security = security_level;
           });

  pybind11::class_<mlir::concretelang::CompilationFeedback>(
      m, "CompilationFeedback")
      .def_readonly("complexity",
                    &mlir::concretelang::CompilationFeedback::complexity)
      .def_readonly("p_error", &mlir::concretelang::CompilationFeedback::pError)
      .def_readonly("global_p_error",
                    &mlir::concretelang::CompilationFeedback::globalPError)
      .def_readonly(
          "total_secret_keys_size",
          &mlir::concretelang::CompilationFeedback::totalSecretKeysSize)
      .def_readonly(
          "total_bootstrap_keys_size",
          &mlir::concretelang::CompilationFeedback::totalBootstrapKeysSize)
      .def_readonly(
          "total_keyswitch_keys_size",
          &mlir::concretelang::CompilationFeedback::totalKeyswitchKeysSize)
      .def_readonly("total_inputs_size",
                    &mlir::concretelang::CompilationFeedback::totalInputsSize)
      .def_readonly("total_output_size",
                    &mlir::concretelang::CompilationFeedback::totalOutputsSize)
      .def_readonly(
          "crt_decompositions_of_outputs",
          &mlir::concretelang::CompilationFeedback::crtDecompositionsOfOutputs);

  pybind11::class_<mlir::concretelang::JitCompilationResult>(
      m, "JITCompilationResult");
  pybind11::class_<mlir::concretelang::JITLambda,
                   std::shared_ptr<mlir::concretelang::JITLambda>>(m,
                                                                   "JITLambda");
  pybind11::class_<JITSupport_Py>(m, "JITSupport")
      .def(pybind11::init([](std::string runtimeLibPath) {
        return jit_support(runtimeLibPath);
      }))
      .def("compile",
           [](JITSupport_Py &support, std::string mlir_program,
              CompilationOptions options) {
             return jit_compile(support, mlir_program.c_str(), options);
           })
      .def("load_client_parameters",
           [](JITSupport_Py &support,
              mlir::concretelang::JitCompilationResult &result) {
             return jit_load_client_parameters(support, result);
           })
      .def("load_compilation_feedback",
           [](JITSupport_Py &support,
              mlir::concretelang::JitCompilationResult &result) {
             return jit_load_compilation_feedback(support, result);
           })
      .def(
          "load_server_lambda",
          [](JITSupport_Py &support,
             mlir::concretelang::JitCompilationResult &result) {
            return jit_load_server_lambda(support, result);
          },
          pybind11::return_value_policy::reference)
      .def("server_call",
           [](JITSupport_Py &support, concretelang::JITLambda &lambda,
              clientlib::PublicArguments &publicArguments,
              clientlib::EvaluationKeys &evaluationKeys) {
             return jit_server_call(support, lambda, publicArguments,
                                    evaluationKeys);
           });

  pybind11::class_<mlir::concretelang::LibraryCompilationResult>(
      m, "LibraryCompilationResult")
      .def(pybind11::init([](std::string outputDirPath, std::string funcname) {
        return mlir::concretelang::LibraryCompilationResult{
            outputDirPath,
            funcname,
        };
      }));
  pybind11::class_<concretelang::serverlib::ServerLambda>(m, "LibraryLambda");
  pybind11::class_<LibrarySupport_Py>(m, "LibrarySupport")
      .def(pybind11::init(
          [](std::string outputPath, std::string runtimeLibraryPath,
             bool generateSharedLib, bool generateStaticLib,
             bool generateClientParameters, bool generateCompilationFeedback,
             bool generateCppHeader) {
            return library_support(
                outputPath.c_str(), runtimeLibraryPath.c_str(),
                generateSharedLib, generateStaticLib, generateClientParameters,
                generateCompilationFeedback, generateCppHeader);
          }))
      .def("compile",
           [](LibrarySupport_Py &support, std::string mlir_program,
              mlir::concretelang::CompilationOptions options) {
             return library_compile(support, mlir_program.c_str(), options);
           })
      .def("load_client_parameters",
           [](LibrarySupport_Py &support,
              mlir::concretelang::LibraryCompilationResult &result) {
             return library_load_client_parameters(support, result);
           })
      .def("load_compilation_feedback",
           [](LibrarySupport_Py &support,
              mlir::concretelang::LibraryCompilationResult &result) {
             return library_load_compilation_feedback(support, result);
           })
      .def(
          "load_server_lambda",
          [](LibrarySupport_Py &support,
             mlir::concretelang::LibraryCompilationResult &result) {
            return library_load_server_lambda(support, result);
          },
          pybind11::return_value_policy::reference)
      .def("server_call",
           [](LibrarySupport_Py &support, serverlib::ServerLambda lambda,
              clientlib::PublicArguments &publicArguments,
              clientlib::EvaluationKeys &evaluationKeys) {
             pybind11::gil_scoped_release release;
             return library_server_call(support, lambda, publicArguments,
                                        evaluationKeys);
           })
      .def("get_shared_lib_path",
           [](LibrarySupport_Py &support) {
             return library_get_shared_lib_path(support);
           })
      .def("get_client_parameters_path", [](LibrarySupport_Py &support) {
        return library_get_client_parameters_path(support);
      });

  class ClientSupport {};
  pybind11::class_<ClientSupport>(m, "ClientSupport")
      .def(pybind11::init())
      .def_static(
          "key_set",
          [](clientlib::ClientParameters clientParameters,
             clientlib::KeySetCache *cache, uint64_t seedMsb,
             uint64_t seedLsb) {
            auto optCache = cache == nullptr
                                ? std::nullopt
                                : std::optional<clientlib::KeySetCache>(*cache);
            return key_set(clientParameters, optCache, seedMsb, seedLsb);
          },
          pybind11::arg().none(false), pybind11::arg().none(true),
          pybind11::arg("seedMsb") = 0, pybind11::arg("seedLsb") = 0)
      .def_static("encrypt_arguments",
                  [](clientlib::ClientParameters clientParameters,
                     clientlib::KeySet &keySet,
                     std::vector<lambdaArgument> args) {
                    std::vector<mlir::concretelang::LambdaArgument *> argsRef;
                    for (auto i = 0u; i < args.size(); i++) {
                      argsRef.push_back(args[i].ptr.get());
                    }
                    return encrypt_arguments(clientParameters, keySet, argsRef);
                  })
      .def_static("decrypt_result", [](clientlib::KeySet &keySet,
                                       clientlib::PublicResult &publicResult) {
        return decrypt_result(keySet, publicResult);
      });
  pybind11::class_<clientlib::KeySetCache>(m, "KeySetCache")
      .def(pybind11::init<std::string &>());

  pybind11::class_<mlir::concretelang::ClientParameters>(m, "ClientParameters")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return clientParametersUnserialize(buffer);
                  })
      .def("serialize",
           [](mlir::concretelang::ClientParameters &clientParameters) {
             return pybind11::bytes(
                 clientParametersSerialize(clientParameters));
           })
      .def("output_signs",
           [](mlir::concretelang::ClientParameters &clientParameters) {
             std::vector<bool> result;
             for (auto output : clientParameters.outputs) {
               if (output.encryption.has_value()) {
                 result.push_back(output.encryption.value().encoding.isSigned);
               } else {
                 result.push_back(true);
               }
             }
             return result;
           })
      .def("input_signs",
           [](mlir::concretelang::ClientParameters &clientParameters) {
             std::vector<bool> result;
             for (auto input : clientParameters.inputs) {
               if (input.encryption.has_value()) {
                 result.push_back(input.encryption.value().encoding.isSigned);
               } else {
                 result.push_back(true);
               }
             }
             return result;
           });

  pybind11::class_<clientlib::KeySet>(m, "KeySet")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    std::unique_ptr<KeySet> result = keySetUnserialize(buffer);
                    return result;
                  })
      .def("serialize",
           [](clientlib::KeySet &keySet) {
             return pybind11::bytes(keySetSerialize(keySet));
           })
      .def("client_parameters",
           [](clientlib::KeySet &keySet) { return keySet.clientParameters(); })
      .def("get_evaluation_keys",
           [](clientlib::KeySet &keySet) { return keySet.evaluationKeys(); });

  pybind11::class_<clientlib::SharedScalarOrTensorData>(m, "Value")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return valueUnserialize(buffer);
                  })
      .def("serialize", [](const clientlib::SharedScalarOrTensorData &value) {
        return pybind11::bytes(valueSerialize(value));
      });

  pybind11::class_<clientlib::ValueExporter>(m, "ValueExporter")
      .def_static("create",
                  [](clientlib::KeySet &keySet,
                     mlir::concretelang::ClientParameters &clientParameters) {
                    return clientlib::ValueExporter(keySet, clientParameters);
                  })
      .def("export_scalar",
           [](clientlib::ValueExporter &exporter, size_t position,
              int64_t value) {
             outcome::checked<clientlib::ScalarOrTensorData, StringError>
                 result = exporter.exportValue(value, position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return clientlib::SharedScalarOrTensorData(
                 std::move(result.value()));
           })
      .def("export_tensor", [](clientlib::ValueExporter &exporter,
                               size_t position, std::vector<int64_t> values,
                               std::vector<int64_t> shape) {
        outcome::checked<clientlib::ScalarOrTensorData, StringError> result =
            exporter.exportValue(values.data(), shape, position);

        if (result.has_error()) {
          throw std::runtime_error(result.error().mesg);
        }

        return clientlib::SharedScalarOrTensorData(std::move(result.value()));
      });

  pybind11::class_<clientlib::ValueDecrypter>(m, "ValueDecrypter")
      .def_static("create",
                  [](clientlib::KeySet &keySet,
                     mlir::concretelang::ClientParameters &clientParameters) {
                    return clientlib::ValueDecrypter(keySet, clientParameters);
                  })
      .def("get_shape",
           [](clientlib::ValueDecrypter &decrypter, size_t position) {
             outcome::checked<std::vector<int64_t>, StringError> result =
                 decrypter.getShape(position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return result.value();
           })
      .def("decrypt_scalar",
           [](clientlib::ValueDecrypter &decrypter, size_t position,
              clientlib::SharedScalarOrTensorData &value) {
             outcome::checked<int64_t, StringError> result =
                 decrypter.decrypt<int64_t>(value.get(), position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return result.value();
           })
      .def("decrypt_tensor",
           [](clientlib::ValueDecrypter &decrypter, size_t position,
              clientlib::SharedScalarOrTensorData &value) {
             outcome::checked<std::vector<int64_t>, StringError> result =
                 decrypter.decryptTensor<int64_t>(value.get(), position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return result.value();
           });

  pybind11::class_<clientlib::PublicArguments,
                   std::unique_ptr<clientlib::PublicArguments>>(
      m, "PublicArguments")
      .def_static(
          "create",
          [](const mlir::concretelang::ClientParameters &clientParameters,
             std::vector<clientlib::SharedScalarOrTensorData> &buffers) {
            return clientlib::PublicArguments(clientParameters, buffers);
          })
      .def_static("deserialize",
                  [](mlir::concretelang::ClientParameters &clientParameters,
                     const pybind11::bytes &buffer) {
                    return publicArgumentsUnserialize(clientParameters, buffer);
                  })
      .def("serialize", [](clientlib::PublicArguments &publicArgument) {
        return pybind11::bytes(publicArgumentsSerialize(publicArgument));
      });
  pybind11::class_<clientlib::PublicResult>(m, "PublicResult")
      .def_static("deserialize",
                  [](mlir::concretelang::ClientParameters &clientParameters,
                     const pybind11::bytes &buffer) {
                    return publicResultUnserialize(clientParameters, buffer);
                  })
      .def("serialize",
           [](clientlib::PublicResult &publicResult) {
             return pybind11::bytes(publicResultSerialize(publicResult));
           })
      .def("n_values",
           [](const clientlib::PublicResult &publicResult) {
             return publicResult.buffers.size();
           })
      .def("get_value",
           [](clientlib::PublicResult &publicResult, size_t position) {
             outcome::checked<clientlib::SharedScalarOrTensorData, StringError>
                 result = publicResult.getValue(position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return result.value();
           });

  pybind11::class_<clientlib::EvaluationKeys>(m, "EvaluationKeys")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return evaluationKeysUnserialize(buffer);
                  })
      .def("serialize", [](clientlib::EvaluationKeys &evaluationKeys) {
        return pybind11::bytes(evaluationKeysSerialize(evaluationKeys));
      });

  pybind11::class_<lambdaArgument>(m, "LambdaArgument")
      .def_static("from_tensor_u8",
                  [](std::vector<uint8_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU8(tensor, dims);
                  })
      .def_static("from_tensor_u16",
                  [](std::vector<uint16_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU16(tensor, dims);
                  })
      .def_static("from_tensor_u32",
                  [](std::vector<uint32_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU32(tensor, dims);
                  })
      .def_static("from_tensor_u64",
                  [](std::vector<uint64_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU64(tensor, dims);
                  })
      .def_static("from_tensor_i8",
                  [](std::vector<int8_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI8(tensor, dims);
                  })
      .def_static("from_tensor_i16",
                  [](std::vector<int16_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI16(tensor, dims);
                  })
      .def_static("from_tensor_i32",
                  [](std::vector<int32_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI32(tensor, dims);
                  })
      .def_static("from_tensor_i64",
                  [](std::vector<int64_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI64(tensor, dims);
                  })
      .def_static("from_scalar", lambdaArgumentFromScalar)
      .def_static("from_signed_scalar", lambdaArgumentFromSignedScalar)
      .def("is_tensor",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsTensor(lambda_arg);
           })
      .def("get_tensor_data",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetTensorData(lambda_arg);
           })
      .def("get_signed_tensor_data",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetSignedTensorData(lambda_arg);
           })
      .def("get_tensor_shape",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetTensorDimensions(lambda_arg);
           })
      .def("is_scalar",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsScalar(lambda_arg);
           })
      .def("is_signed",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsSigned(lambda_arg);
           })
      .def("get_scalar",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetScalar(lambda_arg);
           })
      .def("get_signed_scalar", [](lambdaArgument &lambda_arg) {
        return lambdaArgumentGetSignedScalar(lambda_arg);
      });
}
