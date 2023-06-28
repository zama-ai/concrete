// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_UTILS_H_
#define CONCRETELANG_SUPPORT_UTILS_H_

#include "concretelang/ClientLib/Types.h"
#include <concretelang/ClientLib/ClientLambda.h>
#include <concretelang/ClientLib/KeySet.h>
#include <concretelang/ClientLib/PublicArguments.h>
#include <concretelang/ClientLib/Serializers.h>
#include <concretelang/Runtime/context.h>
#include <concretelang/ServerLib/ServerLambda.h>
#include <concretelang/Support/Error.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <variant>

namespace concretelang {

/// prefix function name with `concrete_` to avoid collision with other function
std::string prefixFuncName(llvm::StringRef funcName);

// construct the function name of the wrapper function that unify function calls
// of compiled circuit
std::string makePackedFunctionName(llvm::StringRef name);

// memref is a struct which is flattened aligned, allocated pointers, offset,
// and two array of rank size for sizes and strides.
uint64_t numArgOfRankedMemrefCallingConvention(uint64_t rank);

template <typename Lambda>
llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
invokeRawOnLambda(Lambda *lambda, clientlib::ClientParameters clientParameters,
                  std::vector<void *> preparedInputArgs,
                  std::optional<clientlib::EvaluationKeys> evaluationKeys,
                  bool simulation = false) {
  // invokeRaw needs to have pointers on arguments and a pointers on the result
  // as last argument.
  // Prepare the outputs vector to store the output value of the lambda.
  auto numOutputs = 0;
  for (auto &output : clientParameters.outputs) {
    auto shape = clientParameters.bufferShape(output, simulation);
    if (shape.size() == 0) {
      // scalar gate
      numOutputs += 1;
    } else {
      // buffer gate
      numOutputs += numArgOfRankedMemrefCallingConvention(shape.size());
    }
  }
  std::vector<uint64_t> outputs(numOutputs);

  // Prepare the raw arguments of invokeRaw, i.e. a vector with pointer on
  // inputs and outputs.
  std::vector<void *> rawArgs(preparedInputArgs.size() +
                              (simulation ? 0 : 1) /*runtime context*/ +
                              1 /* outputs */
  );
  size_t i = 0;
  // Pointers on inputs
  for (auto &arg : preparedInputArgs) {
    rawArgs[i++] = &arg;
  }

  // Some calls require the runtime context while others don't (in simulation)
  std::unique_ptr<mlir::concretelang::RuntimeContext> runtimeContext;
  mlir::concretelang::RuntimeContext *rtCtxPtr;
  if (!simulation) {
    assert(evaluationKeys.has_value() &&
           "evaluation keys are required if not in simulation");
    runtimeContext = std::make_unique<mlir::concretelang::RuntimeContext>(
        evaluationKeys.value());
    // Pointer on runtime context, the rawArgs take pointer on actual value that
    // is passed to the compiled function.
    rtCtxPtr = runtimeContext.get();
    rawArgs[i++] = &rtCtxPtr;
  }

  // Outputs
  rawArgs[i++] = reinterpret_cast<void *>(outputs.data());

  // Invoke
  if (auto err = lambda->invokeRaw(rawArgs)) {
    return std::move(err);
  }

  // Store the result to the PublicResult
  std::vector<clientlib::ScalarOrTensorData> buffers;
  {
    size_t outputOffset = 0;
    for (auto &output : clientParameters.outputs) {
      auto shape = clientParameters.bufferShape(output, simulation);
      if (shape.size() == 0) {
        if (simulation) {
          // value is encrypted (simulated)
          clientlib::ScalarOrTensorData value =
              concretelang::clientlib::ScalarData(
                  outputs[outputOffset++],
                  clientlib::EncryptedScalarElementType,
                  clientlib::EncryptedScalarElementWidth);

          buffers.push_back(std::move(value));
        } else {
          // plain scalar
          clientlib::ScalarOrTensorData value =
              concretelang::clientlib::ScalarData(outputs[outputOffset++],
                                                  output.shape.sign,
                                                  output.shape.width);

          buffers.push_back(std::move(value));
        }
      } else {
        // buffer gate
        auto rank = shape.size();
        auto allocated = (uint64_t *)outputs[outputOffset++];
        auto aligned = (uint64_t *)outputs[outputOffset++];
        auto offset = (size_t)outputs[outputOffset++];
        size_t *sizes = (size_t *)&outputs[outputOffset];
        outputOffset += rank;
        size_t *strides = (size_t *)&outputs[outputOffset];
        outputOffset += rank;

        size_t elementWidth = (output.isEncrypted())
                                  ? clientlib::EncryptedScalarElementWidth
                                  : output.shape.width;

        bool sign = (output.isEncrypted()) ? false : output.shape.sign;

        clientlib::ScalarOrTensorData value =
            clientlib::tensorDataFromMemRef(rank, elementWidth, sign, allocated,
                                            aligned, offset, sizes, strides);

        buffers.push_back(std::move(value));
      }
    }
  }

  return clientlib::PublicResult::fromBuffers(clientParameters, evaluationKeys,
                                              std::move(buffers));
}

template <typename Lambda>
llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
invokeRawOnLambda(Lambda *lambda, clientlib::PublicArguments &arguments,
                  std::optional<clientlib::EvaluationKeys> evaluationKeys,
                  bool simulation = false) {
  // Prepare arguments with the right calling convention
  std::vector<void *> preparedArgs;
  for (auto &arg : arguments.getArguments()) {
    if (std::holds_alternative<clientlib::ScalarData>(arg)) {
      auto scalar = std::get<clientlib::ScalarData>(arg).getValueAsU64();
      preparedArgs.push_back((void *)scalar);
    } else {
      clientlib::TensorData &td = std::get<clientlib::TensorData>(arg);
      // allocated
      preparedArgs.push_back(nullptr);
      // aligned
      preparedArgs.push_back(td.getValuesAsOpaquePointer());
      // offset
      preparedArgs.push_back((void *)0);
      // sizes
      for (size_t size : td.getDimensions()) {
        preparedArgs.push_back((void *)size);
      }

      // Set the stride for each dimension, equal to the product of the
      // following dimensions.
      int64_t stride = td.getNumElements();
      for (size_t size : td.getDimensions()) {
        stride = (size == 0 ? 0 : (stride / size));
        preparedArgs.push_back((void *)stride);
      }
    }
  }
  return invokeRawOnLambda(lambda, arguments.getClientParameters(),
                           preparedArgs, evaluationKeys, simulation);
}

template <typename V, unsigned int N>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const llvm::SmallVector<V, N> vect) {
  OS << "[";
  for (auto v : vect) {
    OS << v << ",";
  }
  OS << "]";
  return OS;
}
} // namespace concretelang

#endif
