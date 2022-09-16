// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <dlfcn.h>

#include "boost/outcome.h"

#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/DynamicArityCall.h"
#include "concretelang/ServerLib/DynamicModule.h"
#include "concretelang/ServerLib/DynamicRankCall.h"
#include "concretelang/ServerLib/ServerLambda.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::CircuitGate;
using concretelang::clientlib::CircuitGateShape;
using concretelang::clientlib::EvaluationKeys;
using concretelang::clientlib::PublicArguments;
using concretelang::clientlib::RuntimeContext;
using concretelang::error::StringError;

outcome::checked<ServerLambda, StringError>
ServerLambda::loadFromModule(std::shared_ptr<DynamicModule> module,
                             std::string funcName) {
  ServerLambda lambda;
  lambda.module =
      module; // prevent module and library handler from being destroyed
  lambda.func =
      (void *(*)(void *, ...))dlsym(module->libraryHandle, funcName.c_str());

  if (auto err = dlerror()) {
    return StringError("Cannot open lambda:") << std::string(err);
  }

  auto param =
      llvm::find_if(module->clientParametersList, [&](ClientParameters param) {
        return param.functionName == funcName;
      });

  if (param == module->clientParametersList.end()) {
    return StringError("cannot find function ")
           << funcName << "in client parameters";
  }

  if (param->outputs.size() != 1) {
    return StringError("ServerLambda: output arity (")
           << std::to_string(param->outputs.size())
           << ") != 1 is not supported";
  }

  lambda.clientParameters = *param;
  return lambda;
}

outcome::checked<ServerLambda, StringError>
ServerLambda::load(std::string funcName, std::string outputPath) {
  OUTCOME_TRY(auto module, DynamicModule::open(outputPath));
  return ServerLambda::loadFromModule(module, funcName);
}

std::unique_ptr<clientlib::PublicResult>
ServerLambda::call(PublicArguments &args, EvaluationKeys &evaluationKeys) {
  std::vector<void *> preparedArgs(args.preparedArgs.begin(),
                                   args.preparedArgs.end());

  RuntimeContext runtimeContext;
  runtimeContext.evaluationKeys = evaluationKeys;
  preparedArgs.push_back((void *)&runtimeContext);

  assert(clientParameters.outputs.size() == 1 &&
         "ServerLambda::call is implemented for only one output");
  auto output = args.clientParameters.outputs[0];
  auto rank = args.clientParameters.bufferShape(output).size();

  // FIXME: Handle sign correctly
  size_t element_width = (output.isEncrypted()) ? 64 : output.shape.width;
  auto result = multi_arity_call_dynamic_rank(func, preparedArgs, rank,
                                              element_width, false);

  std::vector<TensorData> results;
  results.push_back(std::move(result));

  return clientlib::PublicResult::fromBuffers(clientParameters,
                                              std::move(results));
}

} // namespace serverlib
} // namespace concretelang
