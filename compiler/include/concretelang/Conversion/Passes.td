#ifndef CONCRETELANG_CONVERSION_PASSES
#define CONCRETELANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FHETensorOpsToLinalg : Pass<"fhe-tensor-ops-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lowers tensor operations of FHE dialect to linalg.generic";
  let constructor = "mlir::concretelang::createConvertFHETensorOpsToLinalg()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def FHEToTFHEScalar : Pass<"fhe-to-tfhe-scalar", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the FHE dialect to TFHE using the scalar strategy.";
  let description = [{ Lowers operations from the FHE dialect to Std + Math }];
  let constructor = "mlir::concretelang::createConvertFHEToTFHEScalarPass()";
  let options = [];
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def FHEToTFHECrt : Pass<"fhe-to-tfhe-crt", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the FHE dialect to TFHE using the crt strategy.";
  let description = [{ Lowers operations from the FHE dialect to Std + Math }];
  let constructor = "mlir::concretelang::createConvertFHEToTFHECrtPass()";
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

def ExtractSDFGOps : Pass<"extract-sdfg-ops", "::mlir::func::FuncOp"> {
  let summary = "Extracts SDFG ops and creates a static data flow graph";
  let description = [{ Extracts SDFG ops and creates a static data flow graph }];
  let constructor = "mlir::concretelang::createExtractSDFGOps()";
  let dependentDialects = ["mlir::concretelang::SDFG::SDFGDialect"];
}

def BConcreteToCAPI : Pass<"bconcrete-to-capi", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the BConcrete dialect to CAPI calls";
  let description = [{ Lowers operations from the BConcrete dialect to CAPI calls }];
  let constructor = "mlir::concretelang::createConvertBConcreteToCAPIPass()";
  let dependentDialects = ["mlir::concretelang::BConcrete::BConcreteDialect"];
}

def TracingToCAPI : Pass<"tracing-to-capi", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the Tracing dialect to CAPI calls";
  let description = [{ Lowers operations from the Tracing dialect to CAPI calls }];
  let constructor = "mlir::concretelang::createConvertTracingToCAPIPass()";
  let dependentDialects = ["mlir::concretelang::Tracing::TracingDialect"];
}

def SDFGToStreamEmulator : Pass<"sdfg-to-stream-emulator", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the SDFG dialect to Stream Emulator calls";
  let description = [{ Lowers operations from the SDFG dialect to Stream Emulator calls }];
  let constructor = "mlir::concretelang::createConvertSDFGToStreamEmulatorPass()";
  let dependentDialects = ["mlir::concretelang::SDFG::SDFGDialect"];
}

def MLIRLowerableDialectsToLLVM : Pass<"mlir-lowerable-dialects-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers operations from MLIR lowerable dialects to LLVM";
  let constructor = "mlir::concretelang::createConvertMLIRLowerableDialectsToLLVMPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithmeticDialect", "mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
  let options = [];
}

#endif
