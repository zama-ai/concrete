// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
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
using concretelang::clientlib::PublicArguments;
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

  if (!param->outputs[0].encryption.hasValue()) {
    return StringError("ServerLambda: clear output is not yet supported");
  }

  lambda.clientParameters = *param;
  return lambda;
}

outcome::checked<ServerLambda, StringError>
ServerLambda::load(std::string funcName, std::string outputLib) {
  OUTCOME_TRY(auto module, DynamicModule::open(outputLib));
  return ServerLambda::loadFromModule(module, funcName);
}

TensorData dynamicCall(void *(*func)(void *...),
                       std::vector<void *> &preparedArgs, CircuitGate &output) {
  size_t rank = output.shape.dimensions.size();
  return multi_arity_call_dynamic_rank(func, preparedArgs, rank);
}

std::unique_ptr<clientlib::PublicResult>
ServerLambda::call(PublicArguments &args) {
  std::vector<void *> preparedArgs(args.preparedArgs.begin(),
                                   args.preparedArgs.end());
  preparedArgs.push_back((void *)&args.runtimeContext);
  return clientlib::PublicResult::fromBuffers(
      clientParameters,
      {dynamicCall(this->func, preparedArgs, clientParameters.outputs[0])});
  ;
}

} // namespace serverlib
} // namespace concretelang
