#ifndef ZAMALANG_CONVERSION_PASSES
#define ZAMALANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def HLFHETensorOpsToLinalg : FunctionPass<"hlfhe-tensor-ops-to-linalg"> {
  let summary = "Lowers tensor operations of HLFHE dialect to linalg.generic";
  let constructor = "mlir::zamalang::createConvertHLFHETensorOpsToLinalg()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def HLFHEToMidLFHE : Pass<"hlfhe-to-midlfhe", "mlir::ModuleOp"> {
  let summary = "Lowers operations from the HLFHE dialect to MidLFHE";
  let description = [{ Lowers operations from the HLFHE dialect to Std + Math }];
  let constructor = "mlir::zamalang::createConvertHLFHEToMidLFHEPass()";
  let options = [];
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def MLIRLowerableDialectsToLLVM : Pass<"mlir-lowerable-dialects-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers operations from MLIR lowerable dialects to LLVM";
  let constructor = "mlir::zamalang::createConvertMLIRLowerableDialectsToLLVMPass()";
  let dependentDialects = ["mlir::StandardOpsDialect", "mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
  let options = [];
}

#endif
