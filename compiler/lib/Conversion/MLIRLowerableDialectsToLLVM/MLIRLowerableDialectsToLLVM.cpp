#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
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

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"

namespace {
struct MLIRLowerableDialectsToLLVMPass
    : public MLIRLowerableDialectsToLLVMBase<MLIRLowerableDialectsToLLVMPass> {
  void runOnOperation() final;

  /// Convert types to the LLVM dialect-compatible type
  static llvm::Optional<mlir::Type> convertTypes(mlir::Type type);
};
} // namespace

void MLIRLowerableDialectsToLLVMPass::runOnOperation() {
  // Setup the conversion target. We reuse the LLVMConversionTarget that
  // legalize LLVM dialect.
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  // Setup the LLVMTypeConverter (that converts `std` types to `llvm` types) and
  // add our types conversion to `llvm` compatible type.
  mlir::LowerToLLVMOptions options(&getContext());
  options.useBarePtrCallConv = true;
  mlir::LLVMTypeConverter typeConverter(&getContext(), options);
  typeConverter.addConversion(convertTypes);
  typeConverter.addConversion([&](mlir::zamalang::LowLFHE::PlaintextType type) {
    return mlir::IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addConversion([&](mlir::zamalang::LowLFHE::CleartextType type) {
    return mlir::IntegerType::get(type.getContext(), 64);
  });

  // Setup the set of the patterns rewriter. At this point we want to
  // convert the `scf` operations to `std` and `std` operations to `llvm`.
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateLoopToStdConversionPatterns(patterns);
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);

  // Apply a `FullConversion` to `llvm`.
  auto module = getOperation();
  if (mlir::applyFullConversion(module, target, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

llvm::Optional<mlir::Type>
MLIRLowerableDialectsToLLVMPass::convertTypes(mlir::Type type) {
  if (type.isa<mlir::zamalang::LowLFHE::LweCiphertextType>()) {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(type.getContext(), 64));
  }
  return llvm::None;
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
