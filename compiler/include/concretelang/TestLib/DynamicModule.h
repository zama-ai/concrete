// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TESTLIB_DYNAMIC_MODULE_H
#define CONCRETELANG_TESTLIB_DYNAMIC_MODULE_H

#include "concretelang/ClientLib/ClientParameters.h"

namespace mlir {
namespace concretelang {

class DynamicModule {
public:
  ~DynamicModule();
  static llvm::Expected<std::shared_ptr<DynamicModule>>
  open(std::string libraryPath);

private:
  llvm::Error loadClientParametersJSON(std::string path);
  llvm::Error loadSharedLibrary(std::string path);

private:
  std::vector<ClientParameters> clientParametersList;
  void *libraryHandle;

  friend class DynamicLambda;
};

} // namespace concretelang
} // namespace mlir
#endif
