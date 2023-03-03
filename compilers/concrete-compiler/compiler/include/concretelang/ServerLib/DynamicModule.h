// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_DYNAMIC_MODULE_H
#define CONCRETELANG_TESTLIB_DYNAMIC_MODULE_H

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::ClientParameters;
using concretelang::error::StringError;

class DynamicModule {
public:
  ~DynamicModule();

  static outcome::checked<std::shared_ptr<DynamicModule>, StringError>
  open(std::string outputPath);

private:
  outcome::checked<void, StringError>
  loadClientParametersJSON(std::string outputPath);

  outcome::checked<void, StringError> loadSharedLibrary(std::string outputPath);

private:
  std::vector<ClientParameters> clientParametersList;
  void *libraryHandle;

  friend class ServerLambda;
};

} // namespace serverlib
} // namespace concretelang
#endif
