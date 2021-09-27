#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Support/CompilerTools.h"

namespace mlir {
namespace zamalang {

// This is temporary while we doesn't yet have the high-level verification pass
V0FHEConstraint defaultGlobalFHECircuitConstraint{.norm2 = 10, .p = 7};

void initLLVMNativeTarget() {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

void addFilteredPassToPassManager(
    mlir::PassManager &pm, std::unique_ptr<mlir::Pass> pass,
    llvm::function_ref<bool(std::string)> enablePass) {
  if (!enablePass(pass->getArgument().str())) {
    return;
  }
  if (*pass->getOpName() == "module") {
    pm.addPass(std::move(pass));
  } else {
    pm.nest(*pass->getOpName()).addPass(std::move(pass));
  }
};

mlir::LogicalResult CompilerTools::lowerHLFHEToMlirStdsDialect(
    mlir::MLIRContext &context, mlir::Operation *module,
    V0FHEContext &fheContext, LowerOptions options) {
  mlir::PassManager pm(&context);
  if (options.verbose) {
    llvm::errs() << "##################################################\n";
    llvm::errs() << "### HLFHEToMlirStdsDialect pipeline\n";
    context.disableMultithreading();
    pm.enableIRPrinting();
    pm.enableStatistics();
    pm.enableTiming();
    pm.enableVerifier();
  }

  fheContext.constraint = defaultGlobalFHECircuitConstraint;
  fheContext.parameter = *getV0Parameter(fheContext.constraint);
  // Add all passes to lower from HLFHE to LLVM Dialect
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertHLFHETensorOpsToLinalg(),
      options.enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertHLFHEToMidLFHEPass(),
      options.enablePass);
  addFilteredPassToPassManager(
      pm,
      mlir::zamalang::createConvertMidLFHEGlobalParametrizationPass(fheContext),
      options.enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertMidLFHEToLowLFHEPass(),
      options.enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertLowLFHEToConcreteCAPIPass(),
      options.enablePass);

  // Run the passes
  if (pm.run(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult CompilerTools::lowerMlirStdsDialectToMlirLLVMDialect(
    mlir::MLIRContext &context, mlir::Operation *module, LowerOptions options) {
  mlir::PassManager pm(&context);
  if (options.verbose) {
    llvm::errs() << "##################################################\n";
    llvm::errs() << "### MlirStdsDialectToMlirLLVMDialect pipeline\n";
    context.disableMultithreading();
    pm.enableIRPrinting();
    pm.enableStatistics();
    pm.enableTiming();
    pm.enableVerifier();
  }

  // Unparametrize LowLFHE
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertLowLFHEUnparametrizePass(),
      options.enablePass);

  // Bufferize
  addFilteredPassToPassManager(pm, mlir::createTensorConstantBufferizePass(),
                               options.enablePass);
  addFilteredPassToPassManager(pm, mlir::createStdBufferizePass(),
                               options.enablePass);
  addFilteredPassToPassManager(pm, mlir::createTensorBufferizePass(),
                               options.enablePass);
  addFilteredPassToPassManager(pm, mlir::createLinalgBufferizePass(),
                               options.enablePass);
  addFilteredPassToPassManager(pm, mlir::createConvertLinalgToLoopsPass(),
                               options.enablePass);
  addFilteredPassToPassManager(pm, mlir::createFuncBufferizePass(),
                               options.enablePass);
  addFilteredPassToPassManager(pm, mlir::createFinalizingBufferizePass(),
                               options.enablePass);

  // Convert to MLIR LLVM Dialect
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertMLIRLowerableDialectsToLLVMPass(),
      options.enablePass);

  if (pm.run(module).failed()) {
    return mlir::failure();
  }
  return mlir::success();
}

llvm::Expected<std::unique_ptr<llvm::Module>> CompilerTools::toLLVMModule(
    llvm::LLVMContext &llvmContext, mlir::ModuleOp &module,
    llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline) {

  initLLVMNativeTarget();
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return llvm::make_error<llvm::StringError>(
        "failed to translate MLIR to LLVM IR", llvm::inconvertibleErrorCode());
  }

  if (auto err = optPipeline(llvmModule.get())) {
    return llvm::make_error<llvm::StringError>("failed to optimize LLVM IR",
                                               llvm::inconvertibleErrorCode());
  }

  return std::move(llvmModule);
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
  initLLVMNativeTarget();
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
  size_t nbReturn = 0;
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
      if (keySet.inputGate(i).shape.size == 0) {
        // scalar gate
        numInputs = numInputs + 1;
        continue;
      }
      // memref gate, as we follow the standard calling convention
      numInputs = numInputs + 5;
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
      if (gate.shape.size == 0) {
        // scalar gate
        numOutputs = numOutputs + 1;
        continue;
      }
      // memref gate, as we follow the standard calling convention
      numOutputs = numOutputs + 5;
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
  if (info.shape.size != 0) {
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
                                        size_t size) {
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
  if (info.shape.size == 0) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("argument is not a vector: pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (info.shape.size != size) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("vector argument has not the expected size")
            .concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
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
    auto ctBuffer =
        (LweCiphertext_u64 **)malloc(size * sizeof(LweCiphertext_u64 *));
    ciphertextBuffers.push_back(ctBuffer);
    // Allocate ciphertexts and encrypt
    for (auto i = 0; i < size; i++) {
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
  inputs[offset] = (void *)0; // TODO - Better understand how it is used.
  rawArg[offset] = &inputs[offset];
  // aligned
  inputs[offset + 1] = data;
  rawArg[offset + 1] = &inputs[offset + 1];
  // offset
  inputs[offset + 2] = (void *)0;
  rawArg[offset + 2] = &inputs[offset + 2];
  // size
  inputs[offset + 3] = (void *)size;
  rawArg[offset + 3] = &inputs[offset + 3];
  // stride
  inputs[offset + 4] = (void *)0;
  rawArg[offset + 4] = &inputs[offset + 4];
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
  if (info.shape.size == 0) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("output is not a tensor, pos=").concat(llvm::Twine(pos)),
        llvm::inconvertibleErrorCode());
  }
  if (!info.encryption.hasValue()) {
    return llvm::make_error<llvm::StringError>(
        "unencrypted result as tensor output NYI",
        llvm::inconvertibleErrorCode());
  }
  // Get the values as the memref calling convention expect.
  void *allocated = outputs[offset]; // TODO - Better understand how it is used.
  // aligned
  void *aligned = outputs[offset + 1];
  // offset
  size_t offset_r = (size_t)outputs[offset + 2];
  // size
  size_t size_r = (size_t)outputs[offset + 3];
  // stride
  size_t stride = (size_t)outputs[offset + 4];
  // Check the sizes
  if (info.shape.size != size || size_r != size) {
    return llvm::make_error<llvm::StringError>("output bad result buffer size",
                                               llvm::inconvertibleErrorCode());
  }
  // decrypt and fill the result buffer
  for (auto i = 0; i < size_r; i++) {
    LweCiphertext_u64 *ct = ((LweCiphertext_u64 **)(aligned))[i];
    if (auto err = this->keySet.decrypt_lwe(pos, ct, res[i])) {
      return std::move(err);
    }
  }
  return llvm::Error::success();
}

} // namespace zamalang
} // namespace mlir
