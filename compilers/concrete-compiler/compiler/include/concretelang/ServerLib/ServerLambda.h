// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H
#define CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H

#include <cassert>

#include "boost/outcome.h"

#include "concrete-protocol.pb.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/DynamicModule.h"
#include "concretelang/Support/Error.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::ScalarOrTensorData;

/// ServerLambda is a utility class that allows to call a function of a
/// compilation result.
class ServerLambda {

public:
  /// Load the symbol `funcName` from the shared lib in the artifacts folder
  /// located in `outputPath`
  static outcome::checked<ServerLambda, concretelang::error::StringError>
  load(std::string funcName, std::string outputPath);

  /// Load the symbol `funcName` of the dynamic loaded library
  static outcome::checked<ServerLambda, concretelang::error::StringError>
  loadFromModule(std::shared_ptr<DynamicModule> module, std::string funcName);

  /// Call the ServerLambda with public arguments.
  llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
  call(clientlib::PublicArguments &args,
       clientlib::EvaluationKeys &evaluationKeys);

  /// \brief Call the loaded function using opaque pointers to both inputs and
  /// outputs.
  /// \param args Array containing pointers to inputs first, followed by
  /// pointers to outputs.
  /// \return Error if failed, success otherwise.
  llvm::Error invokeRaw(llvm::MutableArrayRef<void *> args);

protected:
  std::unique_ptr<protocol::CircuitInfo> circuitInfo;
  /// holds a pointer to the entrypoint of the shared lib which
  void (*func)(void *...);
  /// Retain module and open shared lib alive
  std::shared_ptr<DynamicModule> module;
};

} // namespace serverlib
} // namespace concretelang

#endif
