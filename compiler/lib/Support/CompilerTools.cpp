#include <llvm/Support/TargetSelect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Support/CompilerTools.h"

namespace mlir {
namespace zamalang {

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

mlir::LogicalResult CompilerTools::lowerHLFHEToMlirLLVMDialect(
    mlir::MLIRContext &context, mlir::Operation *module,
    llvm::function_ref<bool(std::string)> enablePass) {
  mlir::PassManager pm(&context);

  // Add all passes to lower from HLFHE to LLVM Dialect
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertHLFHETensorOpsToLinalg(), enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertHLFHEToMidLFHEPass(), enablePass);
  addFilteredPassToPassManager(
      pm, mlir::zamalang::createConvertMLIRLowerableDialectsToLLVMPass(),
      enablePass);

  // Run the passes
  if (pm.run(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

llvm::Expected<std::unique_ptr<llvm::Module>> CompilerTools::toLLVMModule(
    llvm::LLVMContext &context, mlir::ModuleOp &module,
    llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline) {

  initLLVMNativeTarget();
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(module, context);
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

} // namespace zamalang
} // namespace mlir