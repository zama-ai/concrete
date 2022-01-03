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
using concretelang::clientlib::encrypted_scalars_and_sizes_t;
using concretelang::clientlib::encrypted_scalars_t;

encrypted_scalars_and_sizes_t encrypted_scalars_and_sizes_t_from_MemRef(
    size_t rank, encrypted_scalars_t allocated, encrypted_scalars_t aligned,
    size_t offset, size_t *sizes, size_t *strides);

class ServerLambda {

public:
  static outcome::checked<ServerLambda, concretelang::error::StringError>
  load(std::string funcName, std::string outputLib);

  static outcome::checked<ServerLambda, concretelang::error::StringError>
  loadFromModule(std::shared_ptr<DynamicModule> module, std::string funcName);

  outcome::checked<void, concretelang::error::StringError>
  read_call_write(std::istream &istream, std::ostream &ostream);

protected:
  ClientParameters clientParameters;
  void *(*func)(void *...);
  // Retain module and open shared lib alive
  std::shared_ptr<DynamicModule> module;
};

} // namespace serverlib
} // namespace concretelang

#endif