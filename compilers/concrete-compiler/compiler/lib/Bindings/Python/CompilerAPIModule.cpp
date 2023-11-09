// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Bindings/Python/CompilerAPIModule.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Bindings/Python/CompilerEngine.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Compat.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Dialect/FHE/IR/FHEOpsDialect.h.inc"
#include "concretelang/Support/logging.h"
#include <llvm/Support/Debug.h>
#include <mlir-c/Bindings/Python/Interop.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/OptUtils.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <signal.h>
#include <stdexcept>
#include <string>

using mlir::concretelang::CompilationOptions;
using mlir::concretelang::LambdaArgument;

class SignalGuard {
public:
  SignalGuard() { previousHandler = signal(SIGINT, SignalGuard::handler); }
  ~SignalGuard() { signal(SIGINT, this->previousHandler); }

private:
  void (*previousHandler)(int);

  static void handler(int _signum) {
    llvm::outs() << " Aborting... \n";
    kill(getpid(), SIGKILL);
  }
};

/// Populate the compiler API python module.
void mlir::concretelang::python::populateCompilerAPISubmodule(
    pybind11::module &m) {
  m.doc() = "Concretelang compiler python API";

  m.def("round_trip",
        [](std::string mlir_input) { return roundTrip(mlir_input.c_str()); });

  m.def("set_llvm_debug_flag", [](bool enable) { llvm::DebugFlag = enable; });

  m.def("set_compiler_logging",
        [](bool enable) { mlir::concretelang::setupLogging(enable); });

  m.def("terminate_df_parallelization", &terminateDataflowParallelization);

  m.def("init_df_parallelization", &initDataflowParallelization);

  pybind11::enum_<optimizer::Strategy>(m, "OptimizerStrategy")
      .value("V0", optimizer::Strategy::V0)
      .value("DAG_MONO", optimizer::Strategy::DAG_MONO)
      .value("DAG_MULTI", optimizer::Strategy::DAG_MULTI)
      .export_values();

  pybind11::enum_<concrete_optimizer::Encoding>(m, "Encoding")
      .value("AUTO", concrete_optimizer::Encoding::Auto)
      .value("CRT", concrete_optimizer::Encoding::Crt)
      .value("NATIVE", concrete_optimizer::Encoding::Native)
      .export_values();

  pybind11::class_<CompilationOptions>(m, "CompilationOptions")
      .def(pybind11::init(
          [](std::string funcname) { return CompilationOptions(funcname); }))
      .def("set_funcname",
           [](CompilationOptions &options, std::string funcname) {
             options.mainFuncName = funcname;
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
           })
      .def("set_v0_parameter",
           [](CompilationOptions &options, size_t glweDimension,
              size_t logPolynomialSize, size_t nSmall, size_t brLevel,
              size_t brLogBase, size_t ksLevel, size_t ksLogBase) {
             options.v0Parameter = {glweDimension, logPolynomialSize, nSmall,
                                    brLevel,       brLogBase,         ksLevel,
                                    ksLogBase,     std::nullopt};
           })
      .def("set_v0_parameter",
           [](CompilationOptions &options, size_t glweDimension,
              size_t logPolynomialSize, size_t nSmall, size_t brLevel,
              size_t brLogBase, size_t ksLevel, size_t ksLogBase,
              mlir::concretelang::CRTDecomposition crtDecomposition,
              size_t cbsLevel, size_t cbsLogBase, size_t pksLevel,
              size_t pksLogBase, size_t pksInputLweDimension,
              size_t pksOutputPolynomialSize) {
             mlir::concretelang::PackingKeySwitchParameter pksParam = {
                 pksInputLweDimension, pksOutputPolynomialSize, pksLevel,
                 pksLogBase};
             mlir::concretelang::CitcuitBoostrapParameter crbParam = {
                 cbsLevel, cbsLogBase};
             mlir::concretelang::WopPBSParameter wopPBSParam = {pksParam,
                                                                crbParam};
             mlir::concretelang::LargeIntegerParameter largeIntegerParam = {
                 crtDecomposition, wopPBSParam};
             options.v0Parameter = {glweDimension, logPolynomialSize, nSmall,
                                    brLevel,       brLogBase,         ksLevel,
                                    ksLogBase,     largeIntegerParam};
           })
      .def("force_encoding",
           [](CompilationOptions &options,
              concrete_optimizer::Encoding encoding) {
             options.optimizerConfig.encoding = encoding;
           })
      .def("simulation", [](CompilationOptions &options, bool simulate) {
        options.simulate = simulate;
      });

  pybind11::enum_<mlir::concretelang::PrimitiveOperation>(m,
                                                          "PrimitiveOperation")
      .value("PBS", mlir::concretelang::PrimitiveOperation::PBS)
      .value("WOP_PBS", mlir::concretelang::PrimitiveOperation::WOP_PBS)
      .value("KEY_SWITCH", mlir::concretelang::PrimitiveOperation::KEY_SWITCH)
      .value("CLEAR_ADDITION",
             mlir::concretelang::PrimitiveOperation::CLEAR_ADDITION)
      .value("ENCRYPTED_ADDITION",
             mlir::concretelang::PrimitiveOperation::ENCRYPTED_ADDITION)
      .value("CLEAR_MULTIPLICATION",
             mlir::concretelang::PrimitiveOperation::CLEAR_MULTIPLICATION)
      .value("ENCRYPTED_NEGATION",
             mlir::concretelang::PrimitiveOperation::ENCRYPTED_NEGATION)
      .export_values();

  pybind11::enum_<mlir::concretelang::KeyType>(m, "KeyType")
      .value("SECRET", mlir::concretelang::KeyType::SECRET)
      .value("BOOTSTRAP", mlir::concretelang::KeyType::BOOTSTRAP)
      .value("KEY_SWITCH", mlir::concretelang::KeyType::KEY_SWITCH)
      .value("PACKING_KEY_SWITCH",
             mlir::concretelang::KeyType::PACKING_KEY_SWITCH)
      .export_values();

  pybind11::class_<mlir::concretelang::Statistic>(m, "Statistic")
      .def_readonly("operation", &mlir::concretelang::Statistic::operation)
      .def_readonly("location", &mlir::concretelang::Statistic::location)
      .def_readonly("keys", &mlir::concretelang::Statistic::keys)
      .def_readonly("count", &mlir::concretelang::Statistic::count);

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
          &mlir::concretelang::CompilationFeedback::crtDecompositionsOfOutputs)
      .def_readonly("statistics",
                    &mlir::concretelang::CompilationFeedback::statistics)
      .def_readonly(
          "memory_usage_per_location",
          &mlir::concretelang::CompilationFeedback::memoryUsagePerLoc);

  pybind11::class_<mlir::concretelang::CompilationContext,
                   std::shared_ptr<mlir::concretelang::CompilationContext>>(
      m, "CompilationContext")
      .def(pybind11::init([]() {
        return mlir::concretelang::CompilationContext::createShared();
      }))
      .def("mlir_context",
           [](std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
             auto mlirCtx = cctx->getMLIRContext();
             return pybind11::reinterpret_steal<pybind11::object>(
                 mlirPythonContextToCapsule(wrap(mlirCtx)));
           });

  pybind11::class_<mlir::concretelang::LibraryCompilationResult>(
      m, "LibraryCompilationResult")
      .def(pybind11::init([](std::string outputDirPath, std::string funcname) {
        return mlir::concretelang::LibraryCompilationResult{
            outputDirPath,
            funcname,
        };
      }));
  pybind11::class_<::concretelang::serverlib::ServerLambda>(m, "LibraryLambda");
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
             SignalGuard signalGuard;
             return library_compile(support, mlir_program.c_str(), options);
           })
      .def("compile",
           [](LibrarySupport_Py &support, pybind11::object mlir_module,
              mlir::concretelang::CompilationOptions options,
              std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
             SignalGuard signalGuard;
             return library_compile_module(
                 support,
                 unwrap(mlirPythonCapsuleToModule(mlir_module.ptr())).clone(),
                 options, cctx);
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
             mlir::concretelang::LibraryCompilationResult &result,
             bool useSimulation) {
            return library_load_server_lambda(support, result, useSimulation);
          },
          pybind11::return_value_policy::reference)
      .def("server_call",
           [](LibrarySupport_Py &support,
              ::concretelang::serverlib::ServerLambda lambda,
              ::concretelang::clientlib::PublicArguments &publicArguments,
              ::concretelang::clientlib::EvaluationKeys &evaluationKeys) {
             SignalGuard signalGuard;
             return library_server_call(support, lambda, publicArguments,
                                        evaluationKeys);
           })
      .def("simulate",
           [](LibrarySupport_Py &support,
              ::concretelang::serverlib::ServerLambda lambda,
              ::concretelang::clientlib::PublicArguments &publicArguments) {
             pybind11::gil_scoped_release release;
             return library_simulate(support, lambda, publicArguments);
           })
      .def("get_shared_lib_path",
           [](LibrarySupport_Py &support) {
             return library_get_shared_lib_path(support);
           })
      .def("get_program_info_path", [](LibrarySupport_Py &support) {
        return library_get_program_info_path(support);
      });

  class ClientSupport {};
  pybind11::class_<ClientSupport>(m, "ClientSupport")
      .def(pybind11::init())
      .def_static(
          "key_set",
          [](::concretelang::clientlib::ClientParameters clientParameters,
             ::concretelang::clientlib::KeySetCache *cache, uint64_t seedMsb,
             uint64_t seedLsb) {
            SignalGuard signalGuard;
            auto optCache =
                cache == nullptr
                    ? std::nullopt
                    : std::optional<::concretelang::clientlib::KeySetCache>(
                          *cache);
            return key_set(clientParameters, optCache, seedMsb, seedLsb);
          },
          pybind11::arg().none(false), pybind11::arg().none(true),
          pybind11::arg("seedMsb") = 0, pybind11::arg("seedLsb") = 0)
      .def_static(
          "encrypt_arguments",
          [](::concretelang::clientlib::ClientParameters clientParameters,
             ::concretelang::clientlib::KeySet &keySet,
             std::vector<lambdaArgument> args) {
            std::vector<mlir::concretelang::LambdaArgument *> argsRef;
            for (auto i = 0u; i < args.size(); i++) {
              argsRef.push_back(args[i].ptr.get());
            }
            return encrypt_arguments(clientParameters, keySet, argsRef);
          })
      .def_static(
          "decrypt_result",
          [](::concretelang::clientlib::ClientParameters clientParameters,
             ::concretelang::clientlib::KeySet &keySet,
             ::concretelang::clientlib::PublicResult &publicResult) {
            return decrypt_result(clientParameters, keySet, publicResult);
          });
  pybind11::class_<::concretelang::clientlib::KeySetCache>(m, "KeySetCache")
      .def(pybind11::init<std::string &>());

  pybind11::class_<::concretelang::clientlib::LweSecretKeyParam>(
      m, "LweSecretKeyParam")
      .def("dimension", [](::concretelang::clientlib::LweSecretKeyParam &key) {
        return key.info.asReader().getParams().getLweDimension();
      });

  pybind11::class_<::concretelang::clientlib::BootstrapKeyParam>(
      m, "BootstrapKeyParam")
      .def("input_secret_key_id",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getInputId();
           })
      .def("output_secret_key_id",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getOutputId();
           })
      .def("level",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getLevelCount();
           })
      .def("base_log",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getBaseLog();
           })
      .def("glwe_dimension",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getGlweDimension();
           })
      .def("variance",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getVariance();
           })
      .def("polynomial_size",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getPolynomialSize();
           })
      .def("input_lwe_dimension",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getInputLweDimension();
           });

  pybind11::class_<::concretelang::clientlib::KeyswitchKeyParam>(
      m, "KeyswitchKeyParam")
      .def("input_secret_key_id",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getInputId();
           })
      .def("output_secret_key_id",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getOutputId();
           })
      .def("level",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getLevelCount();
           })
      .def("base_log",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getBaseLog();
           })
      .def("variance", [](::concretelang::clientlib::KeyswitchKeyParam &key) {
        return key.info.asReader().getParams().getVariance();
      });

  pybind11::class_<::concretelang::clientlib::PackingKeyswitchKeyParam>(
      m, "PackingKeyswitchKeyParam")
      .def("input_secret_key_id",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getInputId();
           })
      .def("output_secret_key_id",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getOutputId();
           })
      .def("level",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getLevelCount();
           })
      .def("base_log",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getBaseLog();
           })
      .def("glwe_dimension",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getGlweDimension();
           })
      .def("polynomial_size",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getPolynomialSize();
           })
      .def("input_lwe_dimension",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getInputLweDimension();
           })
      .def("variance",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getVariance();
           });

  pybind11::class_<::concretelang::clientlib::ClientParameters>(
      m, "ClientParameters")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return clientParametersUnserialize(buffer);
                  })
      .def("serialize",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             return pybind11::bytes(
                 clientParametersSerialize(clientParameters));
           })
      .def("output_signs",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             std::vector<bool> result;
             for (auto output : clientParameters.programInfo.asReader()
                                    .getCircuits()[0]
                                    .getOutputs()) {
               if (output.getTypeInfo().hasLweCiphertext() &&
                   output.getTypeInfo()
                       .getLweCiphertext()
                       .getEncoding()
                       .hasInteger()) {
                 result.push_back(output.getTypeInfo()
                                      .getLweCiphertext()
                                      .getEncoding()
                                      .getInteger()
                                      .getIsSigned());
               } else {
                 result.push_back(true);
               }
             }
             return result;
           })
      .def("input_signs",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             std::vector<bool> result;
             for (auto input : clientParameters.programInfo.asReader()
                                   .getCircuits()[0]
                                   .getInputs()) {
               if (input.getTypeInfo().hasLweCiphertext() &&
                   input.getTypeInfo()
                       .getLweCiphertext()
                       .getEncoding()
                       .hasInteger()) {
                 result.push_back(input.getTypeInfo()
                                      .getLweCiphertext()
                                      .getEncoding()
                                      .getInteger()
                                      .getIsSigned());
               } else {
                 result.push_back(true);
               }
             }
             return result;
           })
      .def_readonly("secret_keys",
                    &::concretelang::clientlib::ClientParameters::secretKeys)
      .def_readonly("bootstrap_keys",
                    &::concretelang::clientlib::ClientParameters::bootstrapKeys)
      .def_readonly("keyswitch_keys",
                    &::concretelang::clientlib::ClientParameters::keyswitchKeys)
      .def_readonly(
          "packing_keyswitch_keys",
          &::concretelang::clientlib::ClientParameters::packingKeyswitchKeys);

  pybind11::class_<::concretelang::clientlib::KeySet>(m, "KeySet")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    std::unique_ptr<::concretelang::clientlib::KeySet> result =
                        keySetUnserialize(buffer);
                    return result;
                  })
      .def("serialize",
           [](::concretelang::clientlib::KeySet &keySet) {
             return pybind11::bytes(keySetSerialize(keySet));
           })
      .def("get_evaluation_keys",
           [](::concretelang::clientlib::KeySet &keySet) {
             return ::concretelang::clientlib::EvaluationKeys{
                 keySet.keyset.server};
           });

  pybind11::class_<::concretelang::clientlib::SharedScalarOrTensorData>(m,
                                                                        "Value")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return valueUnserialize(buffer);
                  })
      .def(
          "serialize",
          [](const ::concretelang::clientlib::SharedScalarOrTensorData &value) {
            return pybind11::bytes(valueSerialize(value));
          });

  pybind11::class_<::concretelang::clientlib::ValueExporter>(m, "ValueExporter")
      .def_static(
          "create",
          [](::concretelang::clientlib::KeySet &keySet,
             ::concretelang::clientlib::ClientParameters &clientParameters) {
            return createValueExporter(keySet, clientParameters);
          })
      .def("export_scalar",
           [](::concretelang::clientlib::ValueExporter &exporter,
              size_t position, int64_t value) {
             SignalGuard signalGuard;

             auto info = exporter.circuit.getCircuitInfo()
                             .asReader()
                             .getInputs()[position];
             auto typeTransformer = getPythonTypeTransformer(info);
             auto result = exporter.circuit.prepareInput(
                 typeTransformer({Tensor<int64_t>(value)}), position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return ::concretelang::clientlib::SharedScalarOrTensorData{
                 result.value()};
           })
      .def("export_tensor", [](::concretelang::clientlib::ValueExporter
                                   &exporter,
                               size_t position, std::vector<int64_t> values,
                               std::vector<int64_t> shape) {
        SignalGuard signalGuard;
        std::vector<size_t> dimensions(shape.begin(), shape.end());
        auto info =
            exporter.circuit.getCircuitInfo().asReader().getInputs()[position];
        auto typeTransformer = getPythonTypeTransformer(info);
        auto result = exporter.circuit.prepareInput(
            typeTransformer({Tensor<int64_t>(values, dimensions)}), position);

        if (result.has_error()) {
          throw std::runtime_error(result.error().mesg);
        }

        return ::concretelang::clientlib::SharedScalarOrTensorData{
            result.value()};
      });

  pybind11::class_<::concretelang::clientlib::SimulatedValueExporter>(
      m, "SimulatedValueExporter")
      .def_static(
          "create",
          [](::concretelang::clientlib::ClientParameters &clientParameters) {
            return createSimulatedValueExporter(clientParameters);
          })
      .def("export_scalar",
           [](::concretelang::clientlib::SimulatedValueExporter &exporter,
              size_t position, int64_t value) {
             SignalGuard signalGuard;
             auto info = exporter.circuit.getCircuitInfo()
                             .asReader()
                             .getInputs()[position];
             auto typeTransformer = getPythonTypeTransformer(info);
             auto result = exporter.circuit.prepareInput(
                 typeTransformer({Tensor<int64_t>(value)}), position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return ::concretelang::clientlib::SharedScalarOrTensorData{
                 result.value()};
           })
      .def("export_tensor", [](::concretelang::clientlib::SimulatedValueExporter
                                   &exporter,
                               size_t position, std::vector<int64_t> values,
                               std::vector<int64_t> shape) {
        SignalGuard signalGuard;
        std::vector<size_t> dimensions(shape.begin(), shape.end());
        auto info =
            exporter.circuit.getCircuitInfo().asReader().getInputs()[position];
        auto typeTransformer = getPythonTypeTransformer(info);
        auto result = exporter.circuit.prepareInput(
            typeTransformer({Tensor<int64_t>(values, dimensions)}), position);

        if (result.has_error()) {
          throw std::runtime_error(result.error().mesg);
        }

        return ::concretelang::clientlib::SharedScalarOrTensorData{
            result.value()};
      });

  pybind11::class_<::concretelang::clientlib::ValueDecrypter>(m,
                                                              "ValueDecrypter")
      .def_static(
          "create",
          [](::concretelang::clientlib::KeySet &keySet,
             ::concretelang::clientlib::ClientParameters &clientParameters) {
            return createValueDecrypter(keySet, clientParameters);
          })
      .def("decrypt",
           [](::concretelang::clientlib::ValueDecrypter &decrypter,
              size_t position,
              ::concretelang::clientlib::SharedScalarOrTensorData &value) {
             SignalGuard signalGuard;

             auto result =
                 decrypter.circuit.processOutput(value.value, position);
             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return lambdaArgument{
                 std::make_shared<mlir::concretelang::LambdaArgument>(
                     mlir::concretelang::LambdaArgument{result.value()})};
           });

  pybind11::class_<::concretelang::clientlib::SimulatedValueDecrypter>(
      m, "SimulatedValueDecrypter")
      .def_static(
          "create",
          [](::concretelang::clientlib::ClientParameters &clientParameters) {
            return createSimulatedValueDecrypter(clientParameters);
          })
      .def("decrypt",
           [](::concretelang::clientlib::SimulatedValueDecrypter &decrypter,
              size_t position,
              ::concretelang::clientlib::SharedScalarOrTensorData &value) {
             SignalGuard signalGuard;

             auto result =
                 decrypter.circuit.processOutput(value.value, position);
             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return lambdaArgument{
                 std::make_shared<mlir::concretelang::LambdaArgument>(
                     mlir::concretelang::LambdaArgument{result.value()})};
           });

  pybind11::class_<::concretelang::clientlib::PublicArguments,
                   std::unique_ptr<::concretelang::clientlib::PublicArguments>>(
      m, "PublicArguments")
      .def_static(
          "create",
          [](const ::concretelang::clientlib::ClientParameters
                 &clientParameters,
             std::vector<::concretelang::clientlib::SharedScalarOrTensorData>
                 &buffers) {
            std::vector<TransportValue> vals;
            for (auto buf : buffers) {
              vals.push_back(buf.value);
            }
            return ::concretelang::clientlib::PublicArguments{vals};
          })
      .def_static(
          "deserialize",
          [](::concretelang::clientlib::ClientParameters &clientParameters,
             const pybind11::bytes &buffer) {
            return publicArgumentsUnserialize(clientParameters, buffer);
          })
      .def("serialize",
           [](::concretelang::clientlib::PublicArguments &publicArgument) {
             return pybind11::bytes(publicArgumentsSerialize(publicArgument));
           });
  pybind11::class_<::concretelang::clientlib::PublicResult>(m, "PublicResult")
      .def_static(
          "deserialize",
          [](::concretelang::clientlib::ClientParameters &clientParameters,
             const pybind11::bytes &buffer) {
            return publicResultUnserialize(clientParameters, buffer);
          })
      .def("serialize",
           [](::concretelang::clientlib::PublicResult &publicResult) {
             return pybind11::bytes(publicResultSerialize(publicResult));
           })
      .def("n_values",
           [](const ::concretelang::clientlib::PublicResult &publicResult) {
             return publicResult.values.size();
           })
      .def("get_value",
           [](::concretelang::clientlib::PublicResult &publicResult,
              size_t position) {
             if (position >= publicResult.values.size()) {
               throw std::runtime_error("Failed to get public result value.");
             }
             return ::concretelang::clientlib::SharedScalarOrTensorData{
                 publicResult.values[position]};
           });

  pybind11::class_<::concretelang::clientlib::EvaluationKeys>(m,
                                                              "EvaluationKeys")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return evaluationKeysUnserialize(buffer);
                  })
      .def("serialize",
           [](::concretelang::clientlib::EvaluationKeys &evaluationKeys) {
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
