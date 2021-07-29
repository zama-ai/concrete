#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace {
struct MLIRLowerableDialectsToLLVMPass
    : public MLIRLowerableDialectsToLLVMBase<MLIRLowerableDialectsToLLVMPass> {
  void runOnOperation() final;
};
} // namespace

void MLIRLowerableDialectsToLLVMPass::runOnOperation() {
  // Setup the conversion target. We reuse the LLVMConversionTarget that
  // legalize LLVM dialect.
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  // Setup the LLVMTypeConverter (that converts `std` types to `llvm` types) and
  // add our types conversion to `llvm` compatible type.
  mlir::LLVMTypeConverter typeConverter(&getContext());

  // Setup the set of the patterns rewriter. At this point we want to
  // convert the `scf` operations to `std` and `std` operations to `llvm`.
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateLoopToStdConversionPatterns(patterns);
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Apply a `FullConversion` to `llvm`.
  auto module = getOperation();
  if (mlir::applyFullConversion(module, target, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

namespace mlir {
namespace zamalang {
/// Create a pass for lowering operations the remaining mlir dialects
/// operations, to the LLVM dialect for codegen.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMLIRLowerableDialectsToLLVMPass() {
  return std::make_unique<MLIRLowerableDialectsToLLVMPass>();
}
} // namespace zamalang
} // namespace mlir
