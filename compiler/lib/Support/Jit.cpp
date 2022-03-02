// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "llvm/Support/Error.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include "concretelang/Common/BitsSize.h"
#include <concretelang/Support/Error.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/logging.h>

namespace mlir {
namespace concretelang {

llvm::Expected<std::unique_ptr<JITLambda>>
JITLambda::create(llvm::StringRef name, mlir::ModuleOp &module,
                  llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline,
                  llvm::Optional<llvm::StringRef> runtimeLibPath) {

  // Looking for the function
  auto rangeOps = module.getOps<mlir::LLVM::LLVMFuncOp>();
  auto funcOp = llvm::find_if(rangeOps, [&](mlir::LLVM::LLVMFuncOp op) {
    return op.getName() == name;
  });
  if (funcOp == rangeOps.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find the function to JIT", llvm::inconvertibleErrorCode());
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Create an MLIR execution engine. The execution engine eagerly
  // JIT-compiles the module. If runtimeLibPath is specified, it's passed as a
  // shared library to the JIT compiler.
  std::vector<llvm::StringRef> sharedLibPaths;
  if (runtimeLibPath.hasValue())
    sharedLibPaths.push_back(runtimeLibPath.getValue());
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline,
      /*jitCodeGenOptLevel=*/llvm::None, sharedLibPaths);
  if (!maybeEngine) {
    return StreamStringError("failed to construct the MLIR ExecutionEngine");
  }
  auto &engine = maybeEngine.get();
  auto lambda = std::make_unique<JITLambda>((*funcOp).getType(), name);
  lambda->engine = std::move(engine);

  return std::move(lambda);
}

llvm::Error JITLambda::invokeRaw(llvm::MutableArrayRef<void *> args) {
  auto found = std::find(args.begin(), args.end(), nullptr);
  if (found == args.end()) {
    return this->engine->invokePacked(this->name, args);
  }
  int pos = found - args.begin();
  return StreamStringError("invoke: argument at pos ")
         << pos << " is null or missing";
}

// memref is a struct which is flattened aligned, allocated pointers, offset,
// and two array of rank size for sizes and strides.
uint64_t numArgOfRankedMemrefCallingConvention(uint64_t rank) {
  return 3 + 2 * rank;
}

llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
JITLambda::call(clientlib::PublicArguments &args) {
  // invokeRaw needs to have pointers on arguments and a pointers on the result
  // as last argument.
  // Prepare the outputs vector to store the output value of the lambda.
  auto numOutputs = 0;
  for (auto &output : args.clientParameters.outputs) {
    if (output.shape.dimensions.empty()) {
      // scalar gate
      if (output.encryption.hasValue()) {
        // encrypted scalar : memref<lweSizexi64>
        numOutputs += numArgOfRankedMemrefCallingConvention(1);
      } else {
        // clear scalar
        numOutputs += 1;
      }
    } else {
      // memref gate : rank+1 if the output is encrypted for the lwe size
      // dimension
      auto rank = output.shape.dimensions.size() +
                  (output.encryption.hasValue() ? 1 : 0);
      numOutputs += numArgOfRankedMemrefCallingConvention(rank);
    }
  }
  std::vector<void *> outputs(numOutputs);
  // Prepare the raw arguments of invokeRaw, i.e. a vector with pointer on
  // inputs and outputs.
  std::vector<void *> rawArgs(args.preparedArgs.size() + 1 /*runtime context*/ +
                              outputs.size());
  size_t i = 0;
  // Pointers on inputs
  for (auto &arg : args.preparedArgs) {
    rawArgs[i++] = &arg;
  }
  // Pointer on runtime context, the rawArgs take pointer on actual value that
  // is passed to the compiled function.
  auto rtCtxPtr = &args.runtimeContext;
  rawArgs[i++] = &rtCtxPtr;
  // Pointers on outputs
  for (auto &out : outputs) {
    rawArgs[i++] = &out;
  }

  // Invoke
  if (auto err = invokeRaw(rawArgs)) {
    return std::move(err);
  }

  // Store the result to the PublicResult
  std::vector<clientlib::TensorData> buffers;
  {
    size_t outputOffset = 0;
    for (auto &output : args.clientParameters.outputs) {
      if (output.shape.dimensions.empty() && !output.encryption.hasValue()) {
        // clear scalar
        buffers.push_back(
            clientlib::tensorDataFromScalar((uint64_t)outputs[outputOffset++]));
      } else {
        // encrypted scalar, and tensor gate are memref
        auto rank = output.shape.dimensions.size() +
                    (output.encryption.hasValue() ? 1 : 0);
        auto allocated = (uint64_t *)outputs[outputOffset++];
        auto aligned = (uint64_t *)outputs[outputOffset++];
        auto offset = (size_t)outputs[outputOffset++];
        size_t *sizes = (size_t *)&outputs[outputOffset];
        outputOffset += rank;
        size_t *strides = (size_t *)&outputs[outputOffset];
        outputOffset += rank;
        buffers.push_back(clientlib::tensorDataFromMemRef(
            rank, allocated, aligned, offset, sizes, strides));
      }
    }
  }
  return clientlib::PublicResult::fromBuffers(args.clientParameters, buffers);
}

JITLambda::Argument::Argument(KeySet &keySet) : keySet(keySet) {
  // Setting the inputs
  auto numInputs = 0;
  {
    for (size_t i = 0; i < keySet.numInputs(); i++) {
      auto offset = numInputs;
      auto gate = keySet.inputGate(i);
      inputGates.push_back({gate, offset});
      if (gate.shape.dimensions.empty()) {
        // scalar gate
        if (gate.encryption.hasValue()) {
          // encrypted is a memref<lweSizexi64>
          numInputs = numInputs + numArgOfRankedMemrefCallingConvention(1);
        } else {
          numInputs = numInputs + 1;
        }
        continue;
      }
      // memref gate, as we follow the standard calling convention
      auto rank = keySet.inputGate(i).shape.dimensions.size() +
                  (keySet.isInputEncrypted(i) ? 1 /* for lwe size */ : 0);
      numInputs = numInputs + numArgOfRankedMemrefCallingConvention(rank);
    }
    // Reserve for the context argument
    numInputs = numInputs + 1;
    inputs = std::vector<const void *>(numInputs);
  }

  // Setting the outputs
  {
    auto numOutputs = 0;
    for (size_t i = 0; i < keySet.numOutputs(); i++) {
      auto offset = numOutputs;
      auto gate = keySet.outputGate(i);
      outputGates.push_back({gate, offset});
      if (gate.shape.dimensions.empty()) {
        // scalar gate
        if (gate.encryption.hasValue()) {
          // encrypted is a memref<lweSizexi64>
          numOutputs = numOutputs + numArgOfRankedMemrefCallingConvention(1);
        } else {
          numOutputs = numOutputs + 1;
        }
        continue;
      }
      // memref gate, as we follow the standard calling convention
      auto rank = keySet.outputGate(i).shape.dimensions.size() +
                  (keySet.isOutputEncrypted(i) ? 1 /* for lwe size */ : 0);
      numOutputs = numOutputs + numArgOfRankedMemrefCallingConvention(rank);
    }
    outputs = std::vector<void *>(numOutputs);
  }
  // The raw argument contains pointers to inputs and pointers to store the
  // results
  rawArg = std::vector<void *>(inputs.size() + outputs.size(), nullptr);
  // Set the pointer on outputs on rawArg
  for (auto i = inputs.size(); i < rawArg.size(); i++) {
    rawArg[i] = &outputs[i - inputs.size()];
  }

  // Set the context argument
  keySet.setRuntimeContext(context);
  inputs[numInputs - 1] = &context;
  rawArg[numInputs - 1] = &inputs[numInputs - 1];
}

JITLambda::Argument::~Argument() {
  for (auto ct : allocatedCiphertexts) {
    free(ct);
  }
  for (auto buffer : ciphertextBuffers) {
    free(buffer);
  }
}

llvm::Expected<std::unique_ptr<JITLambda::Argument>>
JITLambda::Argument::create(KeySet &keySet) {
  auto args = std::make_unique<JITLambda::Argument>(keySet);
  return std::move(args);
}

llvm::Error JITLambda::Argument::emitErrorIfTooManyArgs(size_t pos) {
  size_t arity = inputGates.size();
  if (pos < arity) {
    return llvm::Error::success();
  }
  return StreamStringError("The function has arity ")
         << arity << " but is applied to too many arguments";
}

llvm::Error JITLambda::Argument::setArg(size_t pos, uint64_t arg) {
  if (auto error = emitErrorIfTooManyArgs(pos)) {
    return error;
  }
  auto gate = inputGates[pos];
  auto info = std::get<0>(gate);
  auto offset = std::get<1>(gate);

  // Check is the argument is a scalar
  if (!info.shape.dimensions.empty()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument is not a scalar: pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }

  // If argument is not encrypted, just save.
  if (!info.encryption.hasValue()) {
    inputs[offset] = (void *)arg;
    rawArg[offset] = &inputs[offset];
    return llvm::Error::success();
  }
  // Else if is encryted, allocate ciphertext and encrypt.
  uint64_t *ctArg;
  uint64_t ctSize;
  auto check = this->keySet.allocate_lwe(pos, &ctArg, ctSize);
  if (!check) {
    return StreamStringError(check.error().mesg);
  }
  allocatedCiphertexts.push_back(ctArg);
  check = this->keySet.encrypt_lwe(pos, ctArg, arg);
  if (!check) {
    return StreamStringError(check.error().mesg);
  }
  // memref calling convention
  // allocated
  inputs[offset] = nullptr;
  // aligned
  inputs[offset + 1] = ctArg;
  // offset
  inputs[offset + 2] = (void *)0;
  // size
  inputs[offset + 3] = (void *)ctSize;
  // stride
  inputs[offset + 4] = (void *)1;
  rawArg[offset] = &inputs[offset];
  rawArg[offset + 1] = &inputs[offset + 1];
  rawArg[offset + 2] = &inputs[offset + 2];
  rawArg[offset + 3] = &inputs[offset + 3];
  rawArg[offset + 4] = &inputs[offset + 4];
  return llvm::Error::success();
}

llvm::Error JITLambda::Argument::setArg(size_t pos, size_t width,
                                        const void *data,
                                        llvm::ArrayRef<int64_t> shape) {
  if (auto error = emitErrorIfTooManyArgs(pos)) {
    return error;
  }
  auto gate = inputGates[pos];
  auto info = std::get<0>(gate);
  auto offset = std::get<1>(gate);
  // Check if the width is compatible
  // TODO - I found this rules empirically, they are a spec somewhere?
  if (info.shape.width > 64) {
    auto msg = "Bad argument (pos=" + llvm::Twine(pos) + ") : a width of " +
               llvm::Twine(info.shape.width) +
               "bits > 64 is not supported: pos=" + llvm::Twine(pos);
    return llvm::make_error<llvm::StringError>(msg,
                                               llvm::inconvertibleErrorCode());
  }
  auto roundedSize = ::concretelang::common::bitWidthAsWord(info.shape.width);
  if (width != roundedSize) {
    auto msg = "Bad argument (pos=" + llvm::Twine(pos) + ") : expected " +
               llvm::Twine(roundedSize) + "bits" + " but received " +
               llvm::Twine(width) + "bits (rounded from " +
               llvm::Twine(info.shape.width) + ")";
    return llvm::make_error<llvm::StringError>(msg,
                                               llvm::inconvertibleErrorCode());
  }
  // Check the size
  if (info.shape.dimensions.empty()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument is not a vector: pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (shape.size() != info.shape.dimensions.size()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("tensor argument #")
            .concat(llvm::Twine(pos))
            .concat(" has not the expected number of dimension, got ")
            .concat(llvm::Twine(shape.size()))
            .concat(" expected ")
            .concat(llvm::Twine(info.shape.dimensions.size())),
        llvm::inconvertibleErrorCode());
  }
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] != info.shape.dimensions[i]) {
      return llvm::make_error<llvm::StringError>(
          llvm::Twine("tensor argument #")
              .concat(llvm::Twine(pos))
              .concat(" has not the expected dimension #")
              .concat(llvm::Twine(i))
              .concat(" , got ")
              .concat(llvm::Twine(shape[i]))
              .concat(" expected ")
              .concat(llvm::Twine(info.shape.dimensions[i])),
          llvm::inconvertibleErrorCode());
    }
  }
  // If argument is not encrypted, just save with the right calling convention.
  if (info.encryption.hasValue()) {
    // Else if is encrypted
    // For moment we support only 8 bits inputs
    const uint8_t *data8 = (const uint8_t *)data;
    if (width != 8) {
      return llvm::make_error<llvm::StringError>(
          llvm::Twine(
              "argument width > 8 for encrypted gates are not supported: pos=")
              .concat(llvm::Twine(pos)),
          llvm::inconvertibleErrorCode());
    }

    // Allocate a buffer for ciphertexts, the size of the buffer is the number
    // of elements of the tensor * the size of the lwe ciphertext
    auto lweSize = keySet.getInputLweSecretKeyParam(pos).lweSize();
    uint64_t *ctBuffer =
        (uint64_t *)malloc(info.shape.size * lweSize * sizeof(uint64_t));
    ciphertextBuffers.push_back(ctBuffer);
    // Encrypt ciphertexts
    for (size_t i = 0, offset = 0; i < info.shape.size;
         i++, offset += lweSize) {

      auto check = this->keySet.encrypt_lwe(pos, ctBuffer + offset, data8[i]);
      if (!check) {
        return StreamStringError(check.error().mesg);
      }
    }
    // Replace the data by the buffer to ciphertext
    data = (void *)ctBuffer;
  }
  // Set the buffer as the memref calling convention expect.
  // allocated
  inputs[offset] =
      (void *)0; // Indicates that it's not allocated by the MLIR program
  rawArg[offset] = &inputs[offset];
  offset++;
  // aligned
  inputs[offset] = data;
  rawArg[offset] = &inputs[offset];
  offset++;
  // offset
  inputs[offset] = (void *)0;
  rawArg[offset] = &inputs[offset];
  offset++;
  // sizes is an array of size equals to numDim
  for (size_t i = 0; i < shape.size(); i++) {
    inputs[offset] = (void *)shape[i];
    rawArg[offset] = &inputs[offset];
    offset++;
  }
  // If encrypted +1 for the lwe size rank
  if (keySet.isInputEncrypted(pos)) {
    inputs[offset] = (void *)(keySet.getInputLweSecretKeyParam(pos).lweSize());
    rawArg[offset] = &inputs[offset];
    offset++;
  }

  // Set the stride for each dimension, equal to the product of the
  // following dimensions.
  int64_t stride = 1;
  // If encrypted +1 set the stride for the lwe size rank
  if (keySet.isInputEncrypted(pos)) {
    inputs[offset + shape.size()] = (void *)stride;
    rawArg[offset + shape.size()] = &inputs[offset];
    stride *= keySet.getInputLweSecretKeyParam(pos).lweSize();
  }
  for (ssize_t i = shape.size() - 1; i >= 0; i--) {
    inputs[offset + i] = (void *)stride;
    rawArg[offset + i] = &inputs[offset + i];
    stride *= shape[i];
  }
  offset += shape.size();

  return llvm::Error::success();
}

llvm::Error JITLambda::Argument::getResult(size_t pos, uint64_t &res) {
  auto gate = outputGates[pos];
  auto info = std::get<0>(gate);
  auto offset = std::get<1>(gate);

  // Check is the argument is a scalar
  if (info.shape.size != 0) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("output is not a scalar, pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  // If result is not encrypted, just set the result
  if (!info.encryption.hasValue()) {
    res = (uint64_t)(outputs[offset]);
    return llvm::Error::success();
  }
  // Else if is encryted, decrypt
  uint64_t *ct = (uint64_t *)(outputs[offset + 1]);
  auto check = this->keySet.decrypt_lwe(pos, ct, res);
  if (!check) {
    return StreamStringError(check.error().mesg);
  }
  return llvm::Error::success();
}

// Returns the number of elements of the result vector at position
// `pos` or an error if the result is a scalar value
llvm::Expected<size_t> JITLambda::Argument::getResultVectorSize(size_t pos) {
  auto gate = outputGates[pos];
  auto info = std::get<0>(gate);

  if (info.shape.size == 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Result at pos %zu is not a tensor", pos);
  }

  return info.shape.size;
}

// Returns the dimensions of the result tensor at position `pos` or
// an error if the result is a scalar value
llvm::Expected<std::vector<int64_t>>
JITLambda::Argument::getResultDimensions(size_t pos) {
  auto gate = outputGates[pos];
  auto info = std::get<0>(gate);

  if (info.shape.size == 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Result at pos %zu is not a tensor", pos);
  }

  return info.shape.dimensions;
}

llvm::Expected<enum JITLambda::Argument::ResultType>
JITLambda::Argument::getResultType(size_t pos) {
  if (pos >= outputGates.size()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Requesting type for result at index %zu, "
                                   "but lambda only generates %zu results",
                                   pos, outputGates.size());
  }

  auto gate = outputGates[pos];
  auto info = std::get<0>(gate);

  if (info.shape.size == 0) {
    return ResultType::SCALAR;
  } else {
    return ResultType::TENSOR;
  }
}

llvm::Expected<size_t> JITLambda::Argument::getResultWidth(size_t pos) {
  if (pos >= outputGates.size()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Requesting width for result at index %zu, "
                                   "but lambda only generates %zu results",
                                   pos, outputGates.size());
  }

  auto gate = outputGates[pos];
  auto info = std::get<0>(gate);

  // Encrypted values are always returned as 64-bit values for now
  if (info.encryption.hasValue())
    return 64;
  else
    return info.shape.width;
}

llvm::Error JITLambda::Argument::getResult(size_t pos, void *res,
                                           size_t elementSize,
                                           size_t numElements) {

  auto gate = outputGates[pos];
  auto info = std::get<0>(gate);
  auto offset = std::get<1>(gate);

  // Check is the argument is a scalar
  if (info.shape.dimensions.empty()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("output is not a tensor, pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  // Check is the argument is a scalar
  if (info.shape.size != numElements) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("result #")
            .concat(llvm::Twine(pos))
            .concat(" has not the expected size, got ")
            .concat(llvm::Twine(numElements))
            .concat(" expect ")
            .concat(llvm::Twine(info.shape.size)),
        llvm::inconvertibleErrorCode());
  }

  // Get the values as the memref calling convention expect.
  // aligned
  uint8_t *alignedBytes = static_cast<uint8_t *>(outputs[offset + 1]);
  uint8_t *resBytes = static_cast<uint8_t *>(res);
  if (!info.encryption.hasValue()) {
    // just copy values
    for (size_t i = 0; i < numElements; i++) {
      for (size_t j = 0; j < elementSize; j++) {
        *resBytes = *alignedBytes;
        resBytes++;
        alignedBytes++;
      }
    }
  } else {
    // decrypt and fill the result buffer
    auto lweSize = keySet.getOutputLweSecretKeyParam(pos).lweSize();

    for (size_t i = 0, o = 0; i < numElements; i++, o += lweSize) {
      uint64_t *ct = ((uint64_t *)alignedBytes) + o;
      auto check = this->keySet.decrypt_lwe(pos, ct, ((uint64_t *)res)[i]);
      if (!check) {
        return StreamStringError(check.error().mesg);
      }
    }
  }
  return llvm::Error::success();
}

} // namespace concretelang
} // namespace mlir
