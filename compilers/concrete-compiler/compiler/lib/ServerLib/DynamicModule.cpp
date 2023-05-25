// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <dlfcn.h>
#include <fstream>

#include "boost/outcome.h"
#include "concrete-protocol.pb.h"
#include "concretelang/ServerLib/DynamicModule.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/Utils.h"
#include <google/protobuf/message.h>
#include <google/protobuf/stubs/stringpiece.h>
#include <google/protobuf/util/json_util.h>
#include <llvm/Support/Path.h>
#include <memory>

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
DynamicModule::open(std::string outputPath) {
  std::shared_ptr<DynamicModule> module = std::make_shared<DynamicModule>();
  OUTCOME_TRYV(module->loadProgramInfoJSON(outputPath));
  OUTCOME_TRYV(module->loadSharedLibrary(outputPath));
  return module;
}

outcome::checked<void, StringError>
DynamicModule::loadSharedLibrary(std::string outputPath) {
  libraryHandle =
      dlopen(CompilerEngine::Library::getSharedLibraryPath(outputPath).c_str(),
             RTLD_LAZY);
  if (!libraryHandle) {
    return StringError("Cannot open shared library ") << dlerror();
  }
  return outcome::success();
}

outcome::checked<void, StringError>
DynamicModule::loadProgramInfoJSON(std::string outputPath) {
  auto jsonPath = CompilerEngine::Library::getProgramInfoPath(outputPath);
  auto maybeProgramInfo = loadMessageFromJSONFile<protocol::ProgramInfo>(jsonPath);
  if(!maybeProgramInfo){
    return maybeProgramInfo.takeError();
  }
  this->programInfo = *maybeProgramInfo;
  return outcome::success();
}

} // namespace serverlib
} // namespace concretelang
