// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include <dlfcn.h>
#include <fstream>

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"

#include "concretelang/TestLib/DynamicModule.h"

namespace mlir {
namespace concretelang {

DynamicModule::~DynamicModule() {
  if (libraryHandle != nullptr) {
    dlclose(libraryHandle);
  }
}

llvm::Expected<std::shared_ptr<DynamicModule>>
DynamicModule::open(std::string path) {
  std::shared_ptr<DynamicModule> module = std::make_shared<DynamicModule>();
  if (auto err = module->loadClientParametersJSON(path)) {
    return StreamStringError("Cannot load client parameters: ")
           << llvm::toString(std::move(err));
  }
  if (auto err = module->loadSharedLibrary(path)) {
    return StreamStringError("Cannot load client parameters: ")
           << llvm::toString(std::move(err));
  }
  return module;
}

llvm::Error DynamicModule::loadSharedLibrary(std::string path) {
  libraryHandle = dlopen(
      CompilerEngine::Library::getSharedLibraryPath(path).c_str(), RTLD_LAZY);
  if (!libraryHandle) {
    return StreamStringError("Cannot open shared library") << dlerror();
  }
  return llvm::Error::success();
}

llvm::Error DynamicModule::loadClientParametersJSON(std::string path) {

  std::ifstream file(CompilerEngine::Library::getClientParametersPath(path));
  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));
  llvm::Expected<std::vector<ClientParameters>> expectedClientParams =
      llvm::json::parse<std::vector<ClientParameters>>(content);
  if (auto err = expectedClientParams.takeError()) {
    return StreamStringError("Cannot open client parameters: ") << err;
  }
  this->clientParametersList = *expectedClientParams;
  return llvm::Error::success();
}

} // namespace concretelang
} // namespace mlir
