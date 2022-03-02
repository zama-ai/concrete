// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H
#define CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H

#include <cassert>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/DynamicModule.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::encrypted_scalar_t;
using concretelang::clientlib::encrypted_scalars_t;
using concretelang::clientlib::TensorData;

/// ServerLambda is a utility class that allows to call a function of a
/// compilation result.
class ServerLambda {

public:
  /// Load the symbol `funcName` of the compilation result located at the path
  /// `outputLib`.
  static outcome::checked<ServerLambda, concretelang::error::StringError>
  load(std::string funcName, std::string outputLib);

  /// Load the symbol `funcName` of the dynamic loaded library
  static outcome::checked<ServerLambda, concretelang::error::StringError>
  loadFromModule(std::shared_ptr<DynamicModule> module, std::string funcName);

  /// Call the ServerLambda with public arguments.
  std::unique_ptr<clientlib::PublicResult>
  call(clientlib::PublicArguments &args);

protected:
  ClientParameters clientParameters;
  void *(*func)(void *...);
  // Retain module and open shared lib alive
  std::shared_ptr<DynamicModule> module;
};

} // namespace serverlib
} // namespace concretelang

#endif