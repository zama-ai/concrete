#ifndef CONCRETELANG_CONVERSION_PASSES
#define CONCRETELANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FHETensorOpsToLinalg : Pass<"fhe-tensor-ops-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lowers tensor operations of FHE dialect to linalg.generic";
  let constructor = "mlir::concretelang::createConvertFHETensorOpsToLinalg()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def FHEToTFHE : Pass<"fhe-to-tfhe", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the FHE dialect to TFHE";
  let description = [{ Lowers operations from the FHE dialect to Std + Math }];
  let constructor = "mlir::concretelang::createConvertFHEToTFHEPass()";
  let options = [];
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def TFHEGlobalParametrization : Pass<"tfhe-global-parametrization", "mlir::ModuleOp"> {
  let summary = "Inject global fhe parameters to the TFHE dialect";
  let constructor = "mlir::concretelang::createConvertTFHEToConcretePass()";
  let options = [];
  let dependentDialects = ["mlir::concretelang::TFHE::TFHEDialect"];
}

def TFHEToConcrete : Pass<"tfhe-to-concrete", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the TFHE dialect to Concrete";
  let description = [{ Lowers operations from the TFHE dialect to Concrete }];
  let constructor = "mlir::concretelang::createConvertTFHEToConcretePass()";
  let options = [];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::concretelang::TFHE::TFHEDialect"];
}

def LinalgGenericOpWithTensorsToLoops : Pass<"linalg-generic-op-with-tensors-to-loops", "mlir::ModuleOp"> {
  let summary = "Converts linalg.generic ops with tensor inputs / outputs to a loop nest";
  let description = [{ Converts linalg.generic ops with tensor inputs / outputs to a loop nest }];
  let constructor = "mlir::createLinalgGenericOpWithTensorsToLoopsPass()";
  let options = [];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect"];
}

def ConcreteToBConcrete : Pass<"concrete-to-bconcrete", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the Concrete dialect to Bufferized Concrete";
  let description = [{ Lowers operations from the Concrete dialect to Bufferized Concrete }];
  let constructor = "mlir::concretelang::createConvertConcreteToBConcretePass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::concretelang::Concrete::ConcreteDialect", "mlir::concretelang::BConcrete::BConcreteDialect"];
}

def BConcreteToCAPI : Pass<"bconcrete-to-capi", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the BConcrete dialect to CAPI calls";
  let description = [{ Lowers operations from the BConcrete dialect to CAPI calls }];
  let constructor = "mlir::concretelang::createConvertBConcreteToCAPIPass()";
  let dependentDialects = ["mlir::concretelang::BConcrete::BConcreteDialect"];
}

def ConcreteToGPU : Pass<"concrete-to-gpu", "mlir::ModuleOp"> {
  let summary = "Transforms operations in the Concrete dialect to GPU";
  let description = [{ Transforms operations in the Concrete dialect to GPU }];
  let constructor = "mlir::concretelang::createConvertConcreteToGPUPass()";
  let dependentDialects = ["mlir::concretelang::Concrete::ConcreteDialect"];
}

def MLIRLowerableDialectsToLLVM : Pass<"mlir-lowerable-dialects-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers operations from MLIR lowerable dialects to LLVM";
  let constructor = "mlir::concretelang::createConvertMLIRLowerableDialectsToLLVMPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithmeticDialect", "mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
  let options = [];
}

#endif
