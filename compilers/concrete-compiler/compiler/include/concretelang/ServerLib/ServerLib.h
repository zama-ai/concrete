// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H
#define CONCRETELANG_SERVERLIB_SERVER_LAMBDA_H

#include <cassert>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <memory>
#include <vector>
#include <dlfcn.h>

#include "boost/outcome.h"

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Transformers.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/context.h"
#include "concretelang/Support/CompilerEngine.h"

using concretelang::keysets::ServerKeyset;
using concretelang::transformers::ArgTransformer;
using concretelang::transformers::ReturnTransformer;
using concretelang::transformers::TransformerFactory;
using concretelang::values::Value;
using mlir::concretelang::CompilerEngine;
using mlir::concretelang::RuntimeContext;

namespace concretelang {
namespace serverlib {

/// A smart pointer to a dynamic module.
class DynamicModule {
  friend class ServerCircuit;

public:
  ~DynamicModule();
  static Result<std::shared_ptr<DynamicModule>> open(std::string outputPath);

private:
  void *libraryHandle;
};

class ServerCircuit {
  friend class ServerProgram;

public:
  /// Call the circuit with public arguments.
  Result<std::vector<TransportValue>> call(const ServerKeyset &serverKeyset, std::vector<TransportValue> &args);

  Result<std::vector<TransportValue>> simulate(std::vector<TransportValue> &args);
    
  /// Returns the name of this circuit.
  std::string getName();

private:
  ServerCircuit() = default;

  static Result<ServerCircuit>
  fromDynamicModule(const concreteprotocol::CircuitInfo &circuitInfo,
                    std::shared_ptr<DynamicModule> dynamicModule, 
                    bool useSimulation);

  void invoke(RuntimeContext *_runtimeContextPtr);

private:
  concreteprotocol::CircuitInfo circuitInfo;
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
  // std::vector<llvm::MutableArrayRef<void *>> _argRawMaps;
  // std::vector<llvm::ArrayRef<uint64_t>> _returnRawMaps;
  // std::vector<void *> _argRaws;
  // RuntimeContext * _runtimeContextPtr;
  // std::vector<uint64_t> _returnRaws;
  // std::vector<void *> _invocationRaws;
};

/// ServerProgram contains multiple
class ServerProgram {
public:
  /// Loads a server program from a shared lib path essentially.
  static Result<ServerProgram>
  load(const concreteprotocol::ProgramInfo &programInfo, const std::string &sharedLibPath, bool useSimulation);


  Result<ServerCircuit> getServerCircuit(const std::string &circuitName);

private:
  ServerProgram() = default;

private:
  std::vector<ServerCircuit> serverCircuits;
};

} // namespace serverlib
} // namespace concretelang

#endif
