// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H
#define CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H

#include "boost/outcome.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Transformers.h"
#include "concretelang/Common/Values.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <dlfcn.h>
#include <functional>
#include <memory>
#include <vector>

using concretelang::keysets::ServerKeyset;
using concretelang::transformers::ArgTransformer;
using concretelang::transformers::ReturnTransformer;
using concretelang::transformers::TransformerFactory;
using concretelang::values::Value;

namespace concretelang {
namespace serverlib {

/// A smart pointer to a dynamic module.
class DynamicModule {
  friend class ServerCircuit;

public:
  ~DynamicModule();
  static Result<std::shared_ptr<DynamicModule>>
  open(const std::string &outputPath);

private:
  void *libraryHandle;
};

class ServerCircuit {
  friend class ServerProgram;

public:
  static Result<ServerCircuit>
  fromFnPtr(const Message<concreteprotocol::CircuitInfo> &circuitInfo,
            void (*func)(void *...), bool useSimulation);

  /// Call the circuit with public arguments.
  Result<std::vector<TransportValue>>
  call(const ServerKeyset &serverKeyset,
       const std::vector<TransportValue> &args);

  /// Simulate the circuit with public arguments.
  Result<std::vector<TransportValue>>
  simulate(const std::vector<TransportValue> &args);

  /// Returns the name of this circuit.
  std::string getName();

private:
  ServerCircuit() = default;

  static Result<ServerCircuit>
  fromDynamicModule(const Message<concreteprotocol::CircuitInfo> &circuitInfo,
                    std::shared_ptr<DynamicModule> dynamicModule,
                    bool useSimulation);

  void invoke(const ServerKeyset &serverKeyset);

  Message<concreteprotocol::CircuitInfo> circuitInfo;
  bool useSimulation;
  void (*func)(void *...);
  std::shared_ptr<DynamicModule> dynamicModule;
  std::vector<ArgTransformer> argTransformers;
  std::vector<ReturnTransformer> returnTransformers;
  std::vector<Value> argsBuffer;
  std::vector<Value> returnsBuffer;
  std::vector<size_t> argDescriptorSizes;
  std::vector<size_t> returnDescriptorSizes;
  size_t argRawSize;
  size_t returnRawSize;
};

/// ServerProgram contains multiple
class ServerProgram {
public:
  /// Loads a server program from a shared lib path essentially.
  static Result<ServerProgram>
  load(const Message<concreteprotocol::ProgramInfo> &programInfo,
       const std::string &outputPath, bool useSimulation);

  Result<ServerCircuit> getServerCircuit(const std::string &circuitName);

private:
  ServerProgram() = default;

  std::vector<ServerCircuit> serverCircuits;
};

} // namespace serverlib
} // namespace concretelang

#endif
