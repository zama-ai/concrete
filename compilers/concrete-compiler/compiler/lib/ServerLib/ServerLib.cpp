// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <functional>
#include <llvm/ADT/SmallSet.h>
#include <memory>
#include <vector>

#include "boost/outcome.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Transformers.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/context.h"
#include "concretelang/ServerLib/ServerLib.h"
#include "concretelang/Support/CompilerEngine.h"
#include "llvm/ADT/ArrayRef.h"

using concretelang::keysets::ServerKeyset;
using concretelang::transformers::ArgTransformer;
using concretelang::transformers::ReturnTransformer;
using concretelang::transformers::TransformerFactory;
using concretelang::values::Value;
using mlir::concretelang::CompilerEngine;
using mlir::concretelang::RuntimeContext;

namespace concretelang {
namespace serverlib {

// Depending on the strides of the memref, iteration may not be linear in the
// memory space (i.e. it may contain jumps). For this reason we have to compute
// a memory index from the linear index of the iteration space. This structure
// does just that.
struct MultiDimIndexer {
  std::vector<size_t> multiDimensionalIndex;
  size_t offset;
  const std::vector<size_t> &sizes;
  const std::vector<size_t> &strides;

  MultiDimIndexer(size_t offset, const std::vector<size_t> &sizes,
                  const std::vector<size_t> &strides)
      : sizes(sizes), strides(strides) {
    size_t rank = sizes.size();
    this->multiDimensionalIndex.resize(rank);
    for (size_t i = 0; i < rank; i++) {
      this->multiDimensionalIndex[i] = 0;
    }
    // this->sizes = sizes;
    // this->strides = sizes;
    this->offset = offset;
  }

  /// Increments the index.
  void increment() {
    size_t rank = sizes.size();
    for (int r = rank - 1; r >= 0; r--) {
      if (multiDimensionalIndex[r] < sizes[r] - 1) {
        multiDimensionalIndex[r]++;
        return;
      }
      multiDimensionalIndex[r] = 0;
    }
  }

  /// Returns the current index.
  size_t currentIndex() {
    size_t rank = sizes.size();
    size_t g_index = offset;
    size_t default_stride = 1;
    for (int r = rank - 1; r >= 0; r--) {
      g_index += multiDimensionalIndex[r] *
                 ((strides[r] == 0) ? default_stride : strides[r]);
      default_stride *= sizes[r];
    }
    return g_index;
  }
};

// A type representing the memref description of a tensor.
struct MemRefDescriptor {
  size_t precision;
  bool isSigned;
  void *allocated;
  void *aligned;
  size_t offset;
  std::vector<size_t> sizes;
  std::vector<size_t> strides;

  /// Creates a memref descriptor referencing the data contained in a tensor.
  template <typename T> static MemRefDescriptor fromTensor(Tensor<T> &input) {
    std::vector<size_t> strides;
    size_t stride = input.values.size();
    for (size_t dim : input.dimensions) {
      stride = (dim == 0 ? 0 : (stride / dim));
      strides.push_back(stride);
    }
    return MemRefDescriptor{sizeof(T) * 8,
                            std::is_signed<T>(),
                            (void *)nullptr,
                            (void *)input.values.data(),
                            0,
                            input.dimensions,
                            strides};
  }

  /// Creates a memref descriptor from a vector of uint64_t, which is the way to
  /// represent outputs in the current calling convention.
  static MemRefDescriptor fromU64s(llvm::ArrayRef<uint64_t> raw,
                                   size_t precision, bool isSigned) {
    auto rank = (raw.size() - 3) / 2;
    void *allocated = (void *)raw[0];
    void *aligned = (void *)raw[1];
    size_t offset = (size_t)raw[2];
    std::vector<size_t> sizes(rank);
    for (size_t i = 0; i < rank; i++) {
      sizes[i] = (size_t)raw[3 + i];
    }
    std::vector<size_t> strides(rank);
    for (size_t i = 0; i < rank; i++) {
      strides[i] = (size_t)raw[3 + rank + i];
    }
    return MemRefDescriptor{
        precision, isSigned, allocated, aligned, offset, sizes, strides,
    };
  }

  /// Returns the number of elements of the memref.
  size_t getLength() {
    size_t output = 1;
    for (size_t i = 0; i < sizes.size(); i++) {
      output *= sizes[i];
    }
    return output;
  }

  // Allocates a new tensor, and copy the values referenced by a memref
  // descriptor.
  template <typename T> Tensor<T> intoTensor() {
    assert(sizeof(T) * 8 == precision);
    assert(std::is_signed<T>() == isSigned);

    // We create the indexer.
    auto indexer = MultiDimIndexer(offset, sizes, strides);

    // We fill a vector of vales to construct the
    std::vector<T> values(getLength());
    for (size_t i = 0; i < values.size(); i++) {
      T *memrefAligned = reinterpret_cast<T *>(aligned);
      auto index = indexer.currentIndex();
      values[i] = memrefAligned[index];
      indexer.increment();
    }

    return Tensor<T>{values, sizes};
  }

  void intoOpaquePtrs(llvm::MutableArrayRef<void *> &opaquePtrs) {
    opaquePtrs[0] = allocated;
    opaquePtrs[1] = aligned;
    opaquePtrs[2] = (void *)offset;
    for (size_t i = 0; i < sizes.size(); i++) {
      opaquePtrs[3 + i] = (void *)sizes[i];
    }
    for (size_t i = 0; i < strides.size(); i++) {
      opaquePtrs[3 + sizes.size() + i] = (void *)strides[i];
    }
  }
};

struct ScalarDescriptor {
  size_t precision;
  bool isSigned;
  uint64_t val;

  template <typename T> static ScalarDescriptor fromTensor(Tensor<T> &input) {
    T value = input.values[0];
    size_t width = sizeof(T) * 8;
    if (width == 64) {
      return ScalarDescriptor{sizeof(T) * 8, std::is_signed<T>(),
                              (uint64_t)value};
    }
    // Todo : Verify if this is really necessary.
    uint64_t mask = ((uint64_t)1 << width) - 1;
    uint64_t val = ((uint64_t)value) & mask;
    return ScalarDescriptor{sizeof(T) * 8, std::is_signed<T>(), val};
  }

  static ScalarDescriptor fromU64s(llvm::ArrayRef<uint64_t> raw,
                                   size_t precision, bool isSigned) {
    return ScalarDescriptor{precision, isSigned, raw[0]};
  }

  template <typename T> Tensor<T> intoTensor() {
    assert(sizeof(T) * 8 == precision);
    assert(std::is_signed<T>() == isSigned);
    std::vector<T> values{(T)val};
    std::vector<size_t> sizes(0);
    return Tensor<T>(values, sizes);
  }

  void intoOpaquePtrs(llvm::MutableArrayRef<void *> &opaquePtrs) {
    opaquePtrs[0] = (void *)val;
  }
};

/// A type representing an argument used in the invocation of a circuit
/// function.
struct InvocationDescriptor {

  /// An argument can be a memref descriptor, if the argument is a tensor, or a
  /// scalar descriptor, if the argument is a scalar.
  std::variant<MemRefDescriptor, ScalarDescriptor> inner;

  static InvocationDescriptor fromValue(Value &value) {
    if (value.hasElementType<uint8_t>()) {
      return fromTensor<uint8_t>(*value.getTensorPtr<uint8_t>());
    } else if (value.hasElementType<uint16_t>()) {
      return fromTensor<uint16_t>(*value.getTensorPtr<uint16_t>());
    } else if (value.hasElementType<uint32_t>()) {
      return fromTensor<uint32_t>(*value.getTensorPtr<uint32_t>());
    } else if (value.hasElementType<uint64_t>()) {
      return fromTensor<uint64_t>(*value.getTensorPtr<uint64_t>());
    } else if (value.hasElementType<int8_t>()) {
      return fromTensor<int8_t>(*value.getTensorPtr<int8_t>());
    } else if (value.hasElementType<int16_t>()) {
      return fromTensor<int16_t>(*value.getTensorPtr<int16_t>());
    } else if (value.hasElementType<int32_t>()) {
      return fromTensor<int32_t>(*value.getTensorPtr<int32_t>());
    } else if (value.hasElementType<int64_t>()) {
      return fromTensor<int64_t>(*value.getTensorPtr<int64_t>());
    }
    assert(false);
  }

  Value intoValue() {
    if (getIsSigned()) {
      if (getPrecision() == 8) {
        return Value{intoTensor<int8_t>()};
      } else if (getPrecision() == 16) {
        return Value{intoTensor<int16_t>()};
      } else if (getPrecision() == 32) {
        return Value{intoTensor<int32_t>()};
      } else if (getPrecision() == 64) {
        return Value{intoTensor<int64_t>()};
      }
    } else {
      if (getPrecision() == 8) {
        return Value{intoTensor<uint8_t>()};
      } else if (getPrecision() == 16) {
        return Value{intoTensor<uint16_t>()};
      } else if (getPrecision() == 32) {
        return Value{intoTensor<uint32_t>()};
      } else if (getPrecision() == 64) {
        return Value{intoTensor<uint64_t>()};
      }
    }
    assert(false);
  }

  static InvocationDescriptor fromU64s(llvm::ArrayRef<uint64_t> raw,
                                       size_t precision, bool isSigned) {
    if (raw.size() == 1) {
      return InvocationDescriptor{
          ScalarDescriptor::fromU64s(raw, precision, isSigned)};
    } else {
      return InvocationDescriptor{
          MemRefDescriptor::fromU64s(raw, precision, isSigned)};
    }
  }

  void intoOpaquePtrs(llvm::MutableArrayRef<void *> &opaquePtrs) {
    if (std::holds_alternative<ScalarDescriptor>(inner)) {
      std::get<ScalarDescriptor>(inner).intoOpaquePtrs(opaquePtrs);
    } else {
      std::get<MemRefDescriptor>(inner).intoOpaquePtrs(opaquePtrs);
    }
  }

  // Structure used to free memory allocated by the circuit after invocation.
  struct Liberator {

    // Insert in the dropper for further freeing.
    void insert(const InvocationDescriptor &desc) {
      if (std::holds_alternative<MemRefDescriptor>(desc.inner)) {
        ptrs.insert(std::get<MemRefDescriptor>(desc.inner).allocated);
      }
    }

    // Free the memory.
    void tryFree() {
      for (void *ptr : ptrs) {
        if (ptr != nullptr && !isReferenceToMLIRGlobalMemory(ptr)) {
          ::free(ptr);
        }
      }
    }

  private:
    llvm::SmallSet<void *, 8> ptrs;
    static inline bool isReferenceToMLIRGlobalMemory(void *ptr) {
      return reinterpret_cast<uintptr_t>(ptr) == 0xdeadbeef;
    }
  };

private:
  template <typename T>
  static InvocationDescriptor fromTensor(Tensor<T> &tensor) {
    if (tensor.isScalar()) {
      return InvocationDescriptor{ScalarDescriptor::fromTensor(tensor)};
    } else {
      return InvocationDescriptor{MemRefDescriptor::fromTensor(tensor)};
    }
  }

  template <typename T> Tensor<T> intoTensor() {
    if (std::holds_alternative<ScalarDescriptor>(inner)) {
      return std::get<ScalarDescriptor>(inner).intoTensor<T>();
    } else {
      return std::get<MemRefDescriptor>(inner).intoTensor<T>();
    }
  }

  size_t getPrecision() {
    if (std::holds_alternative<ScalarDescriptor>(inner)) {
      return std::get<ScalarDescriptor>(inner).precision;
    } else {
      return std::get<MemRefDescriptor>(inner).precision;
    }
  }

  bool getIsSigned() {
    if (std::holds_alternative<ScalarDescriptor>(inner)) {
      return std::get<ScalarDescriptor>(inner).isSigned;
    } else {
      return std::get<MemRefDescriptor>(inner).isSigned;
    }
  }
};

DynamicModule::~DynamicModule() {
  if (libraryHandle != nullptr) {
    dlclose(libraryHandle);
  }
}

Result<std::shared_ptr<DynamicModule>>
DynamicModule::open(const std::string &sharedLibPath) {
  std::shared_ptr<DynamicModule> module = std::make_shared<DynamicModule>();
  module->libraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY);
  if (!module->libraryHandle) {
    return StringError("Cannot open shared library ") << dlerror();
  }
  mlir::concretelang::dfr::_dfr_register_lib(module->libraryHandle);
  return module;
}

size_t
getGateDescriptionSize(const Message<concreteprotocol::GateInfo> &gateInfo,
                       bool useSimulation) {
  auto shapeToSize = [](concreteprotocol::Shape::Reader shape) -> size_t {
    if (shape.getDimensions().size() == 0) {
      return 1;
    } else {
      return 3 + 2 * shape.getDimensions().size();
    }
  };

  auto typeInfo = gateInfo.asReader().getTypeInfo();

  if (typeInfo.hasIndex()) {
    return shapeToSize(typeInfo.getIndex().getShape());
  } else if (typeInfo.hasPlaintext()) {
    return shapeToSize(typeInfo.getPlaintext().getShape());
  } else if (typeInfo.hasLweCiphertext()) {
    if (useSimulation) {
      if (typeInfo.getLweCiphertext()
              .getConcreteShape()
              .getDimensions()
              .size() == 1) {
        // Initially it was just one ciphertext in native mode. Only an integer
        // will be passed...
        return 1;
      } else {
        // This is either a tensor in native encoding mode, or a tensor in crt
        // mode or whatever. A tensor will be passed, but with the lwe dimension
        // removed basically (hence the -2).
        return shapeToSize(typeInfo.getLweCiphertext().getConcreteShape()) - 2;
      }
    } else {
      return shapeToSize(typeInfo.getLweCiphertext().getConcreteShape());
    }
  } else {
    assert(false);
  }
}

size_t
getGateIntegerPrecision(const Message<concreteprotocol::GateInfo> &gateInfo) {
  if (gateInfo.asReader().getTypeInfo().hasIndex()) {
    return gateInfo.asReader().getTypeInfo().getIndex().getIntegerPrecision();
  } else if (gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return gateInfo.asReader()
        .getTypeInfo()
        .getPlaintext()
        .getIntegerPrecision();
  } else if (gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return gateInfo.asReader()
        .getTypeInfo()
        .getLweCiphertext()
        .getIntegerPrecision();
  }
  assert(false);
}

bool getGateIsSigned(const Message<concreteprotocol::GateInfo> &gateInfo) {
  if (gateInfo.asReader().getTypeInfo().hasIndex()) {
    return gateInfo.asReader().getTypeInfo().getIndex().getIsSigned();
  } else if (gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return gateInfo.asReader().getTypeInfo().getPlaintext().getIsSigned();
  } else if (gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return false;
  }
  assert(false);
}

Result<std::vector<TransportValue>>
ServerCircuit::call(const ServerKeyset &serverKeyset,
                    std::vector<TransportValue> &args) {
  std::vector<TransportValue> returns(returnsBuffer.size());
  mlir::concretelang::dfr::_dfr_register_lib(dynamicModule->libraryHandle);
  if (!mlir::concretelang::dfr::_dfr_is_root_node()) {
    mlir::concretelang::dfr::_dfr_run_remote_scheduler();
    return returns;
  }

  if (args.size() != argsBuffer.size()) {
    return StringError("Called circuit with wrong number of arguments");
  }

  // We load the processed arguments in the args buffer.
  for (size_t i = 0; i < argsBuffer.size(); i++) {
    OUTCOME_TRY(argsBuffer[i], argTransformers[i](args[i]));
  }

  // The arguments has been pushed in the arg buffer, we are now ready to
  // invoke the circuit function.
  invoke(serverKeyset);

  // We process the return values to turn them into transport values.
  for (size_t i = 0; i < returnsBuffer.size(); i++) {
    OUTCOME_TRY(returns[i], returnTransformers[i](returnsBuffer[i]));
  }

  return returns;
}

Result<std::vector<TransportValue>>
ServerCircuit::simulate(std::vector<TransportValue> &args) {
  ServerKeyset emptyKeyset;
  return call(emptyKeyset, args);
}

std::string ServerCircuit::getName() {
  return circuitInfo.asReader().getName();
}

Result<ServerCircuit> ServerCircuit::fromDynamicModule(
    const Message<concreteprotocol::CircuitInfo> &circuitInfo,
    std::shared_ptr<DynamicModule> dynamicModule, bool useSimulation = false) {

  ServerCircuit output;
  output.circuitInfo = circuitInfo;
  output.useSimulation = useSimulation;
  output.dynamicModule = dynamicModule;
  output.func = (void (*)(void *, ...))dlsym(
      dynamicModule->libraryHandle,
      (std::string("_mlir_concrete_") +
       std::string(circuitInfo.asReader().getName().cStr()))
          .c_str());
  if (auto err = dlerror()) {
    return StringError("Circuit symbol not found in dynamic module: ")
           << std::string(err);
  }

  // We prepare the args transformers used to transform transport values into
  // arg values.
  for (auto gateInfo : circuitInfo.asReader().getInputs()) {
    ArgTransformer transformer;
    if (gateInfo.getTypeInfo().hasIndex()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getIndexArgTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasPlaintext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getPlaintextArgTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasLweCiphertext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getLweCiphertextArgTransformer(
                      gateInfo, useSimulation));
    } else {
      return StringError("Malformed input gate info.");
    }
    output.argTransformers.push_back(transformer);
  }

  // We prepare the return transformers used to transform return values into
  // transport values.
  for (auto gateInfo : circuitInfo.asReader().getOutputs()) {
    ReturnTransformer transformer;
    if (gateInfo.getTypeInfo().hasIndex()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getIndexReturnTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasPlaintext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getPlaintextReturnTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasLweCiphertext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getLweCiphertextReturnTransformer(
                      gateInfo, useSimulation));
    } else {
      return StringError("Malformed input gate info.");
    }
    output.returnTransformers.push_back(transformer);
  }

  output.argsBuffer =
      std::vector<Value>(circuitInfo.asReader().getInputs().size());
  output.returnsBuffer =
      std::vector<Value>(circuitInfo.asReader().getOutputs().size());

  output.argRawSize = 0;
  for (auto gateInfo : circuitInfo.asReader().getInputs()) {
    auto descriptorSize = getGateDescriptionSize(gateInfo, useSimulation);
    output.argDescriptorSizes.push_back(descriptorSize);
    output.argRawSize += descriptorSize;
  }

  output.returnRawSize = 0;
  for (auto gateInfo : circuitInfo.asReader().getOutputs()) {
    auto descriptorSize = getGateDescriptionSize(gateInfo, useSimulation);
    output.returnDescriptorSizes.push_back(descriptorSize);
    output.returnRawSize += descriptorSize;
  }

  return output;
}

void ServerCircuit::invoke(const ServerKeyset &serverKeyset) {

  // We create a runtime context from the keyset, and place a pointer to it in
  // the structure.
  RuntimeContext runtimeContext = RuntimeContext(serverKeyset);
  RuntimeContext *_runtimeContextPtr = &runtimeContext;

  auto _argRaws = std::vector<void *>(this->argRawSize);
  auto _argRawMaps = std::vector<llvm::MutableArrayRef<void *>>();
  size_t currentRawIndex = 0;
  for (auto descriptorSize : this->argDescriptorSizes) {
    auto map = llvm::MutableArrayRef<void *>(&_argRaws[currentRawIndex],
                                             descriptorSize);
    _argRawMaps.push_back(map);
    currentRawIndex += descriptorSize;
  }

  auto _returnRaws = std::vector<uint64_t>(this->returnRawSize);
  auto _returnRawMaps = std::vector<llvm::ArrayRef<uint64_t>>();
  currentRawIndex = 0;
  for (auto descriptorSize : this->returnDescriptorSizes) {
    auto map =
        llvm::ArrayRef<uint64_t>(&_returnRaws[currentRawIndex], descriptorSize);
    _returnRawMaps.push_back(map);
    currentRawIndex += descriptorSize;
  }

  auto _invocationRaws = std::vector<void *>();
  for (auto &arg : _argRaws) {
    _invocationRaws.push_back(&arg);
  }
  _invocationRaws.push_back((void *)(&_runtimeContextPtr));
  _invocationRaws.push_back(reinterpret_cast<void *>(_returnRaws.data()));

  // We load the argument descriptors in the _argRaws
  for (unsigned int i = 0; i < circuitInfo.asReader().getInputs().size(); i++) {
    // We construct a descriptor from the input value.
    InvocationDescriptor descriptor =
        InvocationDescriptor::fromValue(argsBuffer[i]);
    // We write the descriptor in the _argRaws via the maps.
    descriptor.intoOpaquePtrs(_argRawMaps[i]);
  }

  func(_invocationRaws.data());

  // The circuit has been executed, we can load the results from the
  // _returnRaws.
  //
  // Note that, the addition of multi outputs made it possible to have aliased
  // outputs. We must then deduplicate the output descriptors before freeing
  // their memory to prevent constructing corrupted outputs and double-freeing.
  auto liberator = InvocationDescriptor::Liberator();
  for (unsigned int i = 0; i < circuitInfo.asReader().getOutputs().size();
       i++) {
    // We read the descriptor from the _returnRaws via the maps.
    size_t precision =
        getGateIntegerPrecision(circuitInfo.asReader().getOutputs()[i]);
    bool isSigned = getGateIsSigned(circuitInfo.asReader().getOutputs()[i]);
    InvocationDescriptor descriptor =
        InvocationDescriptor::fromU64s(_returnRawMaps[i], precision, isSigned);
    // We generate a value from the descriptor which we store in the
    // returnsBuffer.
    returnsBuffer[i] = descriptor.intoValue();
    // // We push the descriptor into the output set for later freeing.
    liberator.insert(descriptor);
  }

  // We (eventually) free the memory allocated for this result by the circuit.
  liberator.tryFree();
}

Result<ServerProgram>
ServerProgram::load(const Message<concreteprotocol::ProgramInfo> &programInfo,
                    const std::string &sharedLibPath, bool useSimulation) {
  ServerProgram output;
  OUTCOME_TRY(auto dynamicModule, DynamicModule::open(sharedLibPath));
  auto sharedDynamicModule = std::shared_ptr<DynamicModule>(dynamicModule);
  std::vector<ServerCircuit> serverCircuits;
  for (auto circuitInfo : programInfo.asReader().getCircuits()) {
    OUTCOME_TRY(auto serverCircuit,
                ServerCircuit::fromDynamicModule(
                    circuitInfo, sharedDynamicModule, useSimulation));
    serverCircuits.push_back(serverCircuit);
  }
  output.serverCircuits = serverCircuits;
  return output;
}

Result<ServerCircuit>
ServerProgram::getServerCircuit(const std::string &circuitName) {
  for (auto serverCircuit : serverCircuits) {
    if (serverCircuit.getName() == circuitName) {
      return serverCircuit;
    }
  }
  return StringError("Tried to get unknown server circuit: `" + circuitName +
                     "`");
}

} // namespace serverlib
} // namespace concretelang
