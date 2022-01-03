// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <dlfcn.h>
#include <fstream>

#include "boost/outcome.h"

#include "concretelang/ServerLib/DynamicModule.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"

namespace concretelang {
namespace serverlib {

using concretelang::error::StringError;
using mlir::concretelang::CompilerEngine;

DynamicModule::~DynamicModule() {
  if (libraryHandle != nullptr) {
    dlclose(libraryHandle);
  }
}

outcome::checked<std::shared_ptr<DynamicModule>, StringError>
DynamicModule::open(std::string libPath) {
  std::shared_ptr<DynamicModule> module = std::make_shared<DynamicModule>();
  OUTCOME_TRYV(module->loadClientParametersJSON(libPath));
  OUTCOME_TRYV(module->loadSharedLibrary(libPath));
  return module;
}

outcome::checked<void, StringError>
DynamicModule::loadSharedLibrary(std::string path) {
  libraryHandle = dlopen(
      CompilerEngine::Library::getSharedLibraryPath(path).c_str(), RTLD_LAZY);
  if (!libraryHandle) {
    return StringError("Cannot open shared library") << dlerror();
  }
  return outcome::success();
}

outcome::checked<void, StringError>
DynamicModule::loadClientParametersJSON(std::string libPath) {
  auto jsonPath = CompilerEngine::Library::getClientParametersPath(libPath);
  OUTCOME_TRY(auto clientParams, ClientParameters::load(jsonPath));
  this->clientParametersList = clientParams;
  return outcome::success();
}

} // namespace serverlib
} // namespace concretelang
