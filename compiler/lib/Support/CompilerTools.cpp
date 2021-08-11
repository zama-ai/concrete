#include <llvm/Support/TargetSelect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Support/CompilerTools.h"

namespace mlir {
namespace zamalang {

// This is temporary while we doesn't yet have the high-level verification pass
FHECircuitConstraint defaultGlobalFHECircuitConstraint{.norm2 = 20, .p = 6};

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
    FHECircuitConstraint &constraint,
    llvm::function_ref<bool(std::string)> enablePass) {
  mlir::PassManager pm(&context);

  // Add all passes to lower from HLFHE to LLVM Dialect
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertHLFHETensorOpsToLinalg(), enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertHLFHEToMidLFHEPass(), enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertLowLFHEToConcreteCAPIPass(), enablePass);
  constraint = defaultGlobalFHECircuitConstraint;

  // Run the passes
  if (pm.run(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult CompilerTools::lowerMlirStdsDialectToMlirLLVMDialect(
    mlir::MLIRContext &context, mlir::Operation *module,
    llvm::function_ref<bool(std::string)> enablePass) {

  mlir::PassManager pm(&context);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertMidLFHEToLowLFHEPass(), enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertMLIRLowerableDialectsToLLVMPass(),
      enablePass);

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
  if (this->type.getNumParams() == args.size() - 1 /*For the result*/) {
    return this->engine->invokePacked(this->name, args);
  }
  return llvm::make_error<llvm::StringError>(
      "wrong number of argument when invoke the JIT lambda",
      llvm::inconvertibleErrorCode());
}

llvm::Error JITLambda::invoke(Argument &args) { return invokeRaw(args.rawArg); }

JITLambda::Argument::Argument(KeySet &keySet) : keySet(keySet) {
  inputs = std::vector<void *>(keySet.numInputs());
  results = std::vector<void *>(keySet.numOutputs());
  // The raw argument contains pointers to inputs and pointers to store the
  // results
  rawArg =
      std::vector<void *>(keySet.numInputs() + keySet.numOutputs(), nullptr);
  // Set the results pointer on the rawArg
  for (auto i = keySet.numInputs(); i < rawArg.size(); i++) {
    rawArg[i] = &results[i - keySet.numInputs()];
  }
}

JITLambda::Argument::~Argument() {
  int err;
  for (auto i = 0; i < keySet.numInputs(); i++) {
    if (keySet.isInputEncrypted(i)) {
      free_lwe_ciphertext_u64(&err, (LweCiphertext_u64 *)(inputs[i]));
    }
  }
}

llvm::Expected<std::unique_ptr<JITLambda::Argument>>
JITLambda::Argument::create(KeySet &keySet) {
  auto args = std::make_unique<JITLambda::Argument>(keySet);
  return std::move(args);
}

llvm::Error JITLambda::Argument::setArg(size_t pos, uint64_t arg) {
  // If argument is not encrypted, just save.
  if (!keySet.isInputEncrypted(pos)) {
    inputs[pos] = (void *)arg;
    rawArg[pos] = &inputs[pos];
    return llvm::Error::success();
  }
  // Else if is encryted, allocate ciphertext.
  LweCiphertext_u64 *ctArg;
  if (auto err = this->keySet.allocate_lwe(pos, &ctArg)) {
    return std::move(err);
  }
  if (auto err = this->keySet.encrypt_lwe(pos, ctArg, arg)) {
    return std::move(err);
  }
  inputs[pos] = ctArg;
  rawArg[pos] = &inputs[pos];
  return llvm::Error::success();
}

llvm::Error JITLambda::Argument::getResult(size_t pos, uint64_t &res) {
  // If result is not encrypted, just set the result
  if (!keySet.isOutputEncrypted(pos)) {
    res = (uint64_t)(results[pos]);
    return llvm::Error::success();
  }
  // Else if is encryted, decrypt
  LweCiphertext_u64 *ct = (LweCiphertext_u64 *)(results[pos]);
  if (auto err = this->keySet.decrypt_lwe(pos, ct, res)) {
    return std::move(err);
  }
  return llvm::Error::success();
}

} // namespace zamalang
} // namespace mlir