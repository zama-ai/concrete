#ifndef CONCRETELANG_CONVERSION_PASSES
#define CONCRETELANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FHETensorOpsToLinalg : FunctionPass<"fhe-tensor-ops-to-linalg"> {
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

def ConcreteToBConcrete : Pass<"concrete-to-bconcrete", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the Concrete dialect to Bufferized Concrete";
  let description = [{ Lowers operations from the Concrete dialect to Bufferized Concrete }];
  let constructor = "mlir::concretelang::createConvertConcreteToBConcretePass()";
  let options = [];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::concretelang::Concrete::ConcreteDialect", "mlir::concretelang::BConcrete::BConcreteDialect"];
}

def ConcreteToConcreteCAPI : Pass<"concrete-to-concrete-c-api", "mlir::ModuleOp"> {
  let summary = "Lower operations from the Concrete dialect to std with function call to the Concrete C API";
  let constructor = "mlir::concretelang::createConvertConcreteToConcreteCAPIPass()";
  let dependentDialects = ["mlir::concretelang::Concrete::ConcreteDialect", "mlir::StandardOpsDialect", "mlir::memref::MemRefDialect"];
}

def BConcreteToBConcreteCAPI : Pass<"bconcrete-to-bconcrete-c-api", "mlir::ModuleOp"> {
  let summary = "Lower operations from the Bufferized Concrete dialect to std with function call to the Bufferized Concrete C API";
  let constructor = "mlir::concretelang::createConvertBConcreteToBConcreteCAPIPass()";
  let dependentDialects = ["mlir::concretelang::BConcrete::BConcreteDialect", "mlir::StandardOpsDialect", "mlir::memref::MemRefDialect"];
}

def ConcreteUnparametrize : Pass<"concrete-unparametrize", "mlir::ModuleOp"> {
  let summary = "Unparametrize Concrete types and remove unrealized_conversion_cast";
  let constructor = "mlir::concretelang::createConvertConcreteToConcreteCAPIPass()";
  let dependentDialects = ["mlir::concretelang::Concrete::ConcreteDialect", "mlir::StandardOpsDialect", "mlir::memref::MemRefDialect"];
}

def MLIRLowerableDialectsToLLVM : Pass<"mlir-lowerable-dialects-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers operations from MLIR lowerable dialects to LLVM";
  let constructor = "mlir::concretelang::createConvertMLIRLowerableDialectsToLLVMPass()";
  let dependentDialects = ["mlir::StandardOpsDialect", "mlir::arith::ArithmeticDialect", "mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
  let options = [];
}

#endif
