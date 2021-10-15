#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <zamalang/Support/Jit.h>
#include <zamalang/Support/logging.h>

namespace mlir {
namespace zamalang {

// JIT-compiles `module` invokes `func` with the arguments passed in
// `jitArguments` and `keySet`
mlir::LogicalResult
runJit(mlir::ModuleOp module, llvm::StringRef func,
       llvm::ArrayRef<uint64_t> funcArgs, mlir::zamalang::KeySet &keySet,
       std::function<llvm::Error(llvm::Module *)> optPipeline,
       llvm::raw_ostream &os) {
  // Create the JIT lambda
  auto maybeLambda =
      mlir::zamalang::JITLambda::create(func, module, optPipeline);
  if (!maybeLambda) {
    return mlir::failure();
  }
  auto lambda = std::move(maybeLambda.get());

  // Create the arguments of the JIT lambda
  auto maybeArguments = mlir::zamalang::JITLambda::Argument::create(keySet);
  if (auto err = maybeArguments.takeError()) {
    ::mlir::zamalang::log_error()
        << "Cannot create lambda arguments: " << err << "\n";
    llvm::consumeError(std::move(err));
    return mlir::failure();
  }

  // Set the arguments
  auto arguments = std::move(maybeArguments.get());
  for (size_t i = 0; i < funcArgs.size(); i++) {
    if (auto err = arguments->setArg(i, funcArgs[i])) {
      ::mlir::zamalang::log_error()
          << "Cannot push argument " << i << ": " << err << "\n";
      llvm::consumeError(std::move(err));
      return mlir::failure();
    }
  }
  // Invoke the lambda
  if (auto err = lambda->invoke(*arguments)) {
    ::mlir::zamalang::log_error() << "Cannot invoke : " << err << "\n";
    llvm::consumeError(std::move(err));
    return mlir::failure();
  }
  uint64_t res = 0;
  if (auto err = arguments->getResult(0, res)) {
    ::mlir::zamalang::log_error() << "Cannot get result : " << err << "\n";
    llvm::consumeError(std::move(err));
    return mlir::failure();
  }
  llvm::errs() << res << "\n";
  return mlir::success();
}

llvm::Expected<std::unique_ptr<JITLambda>>
JITLambda::create(llvm::StringRef name, mlir::ModuleOp &module,
                  llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline) {

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
  // JIT-compiles the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  if (!maybeEngine) {
    return llvm::make_error<llvm::StringError>(
        "failed to construct the MLIR ExecutionEngine",
        llvm::inconvertibleErrorCode());
  }
  auto &engine = maybeEngine.get();
  auto lambda = std::make_unique<JITLambda>((*funcOp).getType(), name);
  lambda->engine = std::move(engine);

  return std::move(lambda);
}

llvm::Error JITLambda::invokeRaw(llvm::MutableArrayRef<void *> args) {
  // size_t nbReturn = 0;
  // TODO - This check break with memref as we have 5 returns args.
  // if (!this->type.getReturnType().isa<mlir::LLVM::LLVMVoidType>()) {
  // nbReturn = 1;
  // }
  // if (this->type.getNumParams() != args.size() - nbReturn) {
  //   return llvm::make_error<llvm::StringError>(
  //       "invokeRaw: wrong number of argument",
  //       llvm::inconvertibleErrorCode());
  // }
  if (llvm::find(args, nullptr) != args.end()) {
    return llvm::make_error<llvm::StringError>(
        "invoke: some arguments are null", llvm::inconvertibleErrorCode());
  }
  return this->engine->invokePacked(this->name, args);
}

llvm::Error JITLambda::invoke(Argument &args) {
  return std::move(invokeRaw(args.rawArg));
}

JITLambda::Argument::Argument(KeySet &keySet) : keySet(keySet) {
  // Setting the inputs
  {
    auto numInputs = 0;
    for (size_t i = 0; i < keySet.numInputs(); i++) {
      auto offset = numInputs;
      auto gate = keySet.inputGate(i);
      inputGates.push_back({gate, offset});
      if (keySet.inputGate(i).shape.dimensions.empty()) {
        // scalar gate
        numInputs = numInputs + 1;
        continue;
      }
      // memref gate, as we follow the standard calling convention
      numInputs = numInputs + 3;
      // Offsets and strides are array of size N where N is the number of
      // dimension of the tensor.
      numInputs = numInputs + 2 * keySet.inputGate(i).shape.dimensions.size();
    }
    inputs = std::vector<void *>(numInputs);
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
        numOutputs = numOutputs + 1;
        continue;
      }
      // memref gate, as we follow the standard calling convention
      numOutputs = numOutputs + 3;
      // Offsets and strides are array of size N where N is the number of
      // dimension of the tensor.
      numOutputs =
          numOutputs + 2 * keySet.outputGate(i).shape.dimensions.size();
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

  // Setup runtime context with appropriate keys
  keySet.initGlobalRuntimeContext();
}

JITLambda::Argument::~Argument() {
  int err;
  for (auto ct : allocatedCiphertexts) {
    free_lwe_ciphertext_u64(&err, ct);
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

llvm::Error JITLambda::Argument::setArg(size_t pos, uint64_t arg) {
  if (pos >= inputGates.size()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument index out of bound: pos=")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
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
  LweCiphertext_u64 *ctArg;
  if (auto err = this->keySet.allocate_lwe(pos, &ctArg)) {
    return std::move(err);
  }
  allocatedCiphertexts.push_back(ctArg);
  if (auto err = this->keySet.encrypt_lwe(pos, ctArg, arg)) {
    return std::move(err);
  }
  inputs[offset] = ctArg;
  rawArg[offset] = &inputs[offset];
  return llvm::Error::success();
}

llvm::Error JITLambda::Argument::setArg(size_t pos, size_t width, void *data,
                                        size_t numDim, const size_t *dims) {
  auto gate = inputGates[pos];
  auto info = std::get<0>(gate);
  auto offset = std::get<1>(gate);
  // Check if the width is compatible
  // TODO - I found this rules empirically, they are a spec somewhere?
  if (info.shape.width <= 8 && width != 8) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument width should be 8: pos=")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (info.shape.width > 8 && info.shape.width <= 16 && width != 16) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument width should be 16: pos=")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (info.shape.width > 16 && info.shape.width <= 32 && width != 32) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument width should be 32: pos=")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (info.shape.width > 32 && info.shape.width <= 64 && width != 64) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument width should be 64: pos=")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (info.shape.width > 64) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument width not supported: pos=")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  // Check the size
  if (info.shape.dimensions.empty()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument is not a vector: pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (numDim != info.shape.dimensions.size()) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("tensor argument #")
            .concat(llvm::Twine(pos))
            .concat(" has not the expected number of dimension, got ")
            .concat(llvm::Twine(numDim))
            .concat(" expected ")
            .concat(llvm::Twine(info.shape.dimensions.size())),
        llvm::inconvertibleErrorCode());
  }
  for (size_t i = 0; i < numDim; i++) {
    if (dims[i] != info.shape.dimensions[i]) {
      return llvm::make_error<llvm::StringError>(
          llvm::Twine("tensor argument #")
              .concat(llvm::Twine(pos))
              .concat(" has not the expected dimension #")
              .concat(llvm::Twine(i))
              .concat(" , got ")
              .concat(llvm::Twine(dims[i]))
              .concat(" expected ")
              .concat(llvm::Twine(info.shape.dimensions[i])),
          llvm::inconvertibleErrorCode());
    }
  }
  // If argument is not encrypted, just save with the right calling convention.
  if (info.encryption.hasValue()) {
    // Else if is encrypted
    // For moment we support only 8 bits inputs
    uint8_t *data8 = (uint8_t *)data;
    if (width != 8) {
      return llvm::make_error<llvm::StringError>(
          llvm::Twine(
              "argument width > 8 for encrypted gates are not supported: pos=")
              .concat(llvm::Twine(pos)),
          llvm::inconvertibleErrorCode());
    }

    // Allocate a buffer for ciphertexts.
    auto ctBuffer = (LweCiphertext_u64 **)malloc(info.shape.size *
                                                 sizeof(LweCiphertext_u64 *));
    ciphertextBuffers.push_back(ctBuffer);
    // Allocate ciphertexts and encrypt
    for (size_t i = 0; i < info.shape.size; i++) {
      if (auto err = this->keySet.allocate_lwe(pos, &ctBuffer[i])) {
        return std::move(err);
      }
      allocatedCiphertexts.push_back(ctBuffer[i]);
      if (auto err = this->keySet.encrypt_lwe(pos, ctBuffer[i], data8[i])) {
        return std::move(err);
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
  for (size_t i = 0; i < numDim; i++) {
    inputs[offset] = (void *)dims[i];
    rawArg[offset] = &inputs[offset];
    offset++;
  }
  // strides is an array of size equals to numDim
  for (size_t i = 0; i < numDim; i++) {
    inputs[offset] = (void *)0;
    rawArg[offset] = &inputs[offset];
    offset++;
  }
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
  LweCiphertext_u64 *ct = (LweCiphertext_u64 *)(outputs[offset]);
  if (auto err = this->keySet.decrypt_lwe(pos, ct, res)) {
    return std::move(err);
  }
  return llvm::Error::success();
}

llvm::Error JITLambda::Argument::getResult(size_t pos, uint64_t *res,
                                           size_t size) {

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
  if (info.shape.size != size) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("result #")
            .concat(llvm::Twine(pos))
            .concat(" has not the expected size, got ")
            .concat(llvm::Twine(size))
            .concat(" expect ")
            .concat(llvm::Twine(info.shape.size)),
        llvm::inconvertibleErrorCode());
  }

  // Get the values as the memref calling convention expect.
  // aligned
  void *allocated = outputs[offset];
  void *aligned = outputs[offset + 1];
  if (!info.encryption.hasValue()) {
    // just copy values
    for (size_t i = 0; i < size; i++) {
      res[i] = ((uint64_t *)(aligned))[i];
    }
  } else {
    // decrypt and fill the result buffer
    for (size_t i = 0; i < size; i++) {
      LweCiphertext_u64 *ct = ((LweCiphertext_u64 **)(aligned))[i];
      if (auto err = this->keySet.decrypt_lwe(pos, ct, res[i])) {
        return std::move(err);
      }
    }
  }
  return llvm::Error::success();
}

} // namespace zamalang
} // namespace mlir
