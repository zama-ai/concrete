// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <dlfcn.h>

#include "boost/outcome.h"

#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/DynamicModule.h"
#include "concretelang/ServerLib/ServerLambda.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Utils.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::CircuitGate;
using concretelang::clientlib::CircuitGateShape;
using concretelang::clientlib::EvaluationKeys;
using concretelang::clientlib::PublicArguments;
using concretelang::error::StringError;
using mlir::concretelang::StreamStringError;

outcome::checked<ServerLambda, StringError>
ServerLambda::loadFromModule(std::shared_ptr<DynamicModule> module,
                             std::string funcName) {
  auto packedFuncName = ::concretelang::makePackedFunctionName(funcName);
  ServerLambda lambda;
  lambda.module =
      module; // prevent module and library handler from being destroyed
  lambda.func = (void (*)(void *, ...))dlsym(module->libraryHandle,
                                             packedFuncName.c_str());

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

llvm::Error ServerLambda::invokeRaw(llvm::MutableArrayRef<void *> args) {
  auto found = std::find(args.begin(), args.end(), nullptr);
  if (found == args.end()) {
    assert(func != nullptr && "func pointer shouldn't be null");
    func(args.data());
    return llvm::Error::success();
  }
  int pos = found - args.begin();
  return StreamStringError("invoke: argument at pos ")
         << pos << " is null or missing";
}

llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
ServerLambda::call(PublicArguments &args, EvaluationKeys &evaluationKeys) {
  return invokeRawOnLambda(this, args.clientParameters, args.preparedArgs,
                           evaluationKeys);
}

} // namespace serverlib
} // namespace concretelang
