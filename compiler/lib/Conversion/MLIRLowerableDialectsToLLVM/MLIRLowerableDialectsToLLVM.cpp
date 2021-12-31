// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/RT/Analysis/Autopar.h"
#include "concretelang/Dialect/RT/IR/RTTypes.h"

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
  target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

  // Setup the LLVMTypeConverter (that converts `std` types to `llvm` types) and
  // add our types conversion to `llvm` compatible type.
  mlir::LowerToLLVMOptions options(&getContext());
  mlir::LLVMTypeConverter typeConverter(&getContext(), options);
  typeConverter.addConversion(convertTypes);
  typeConverter.addConversion(
      [&](mlir::concretelang::Concrete::PlaintextType type) {
        return mlir::IntegerType::get(type.getContext(), 64);
      });
  typeConverter.addConversion(
      [&](mlir::concretelang::Concrete::CleartextType type) {
        return mlir::IntegerType::get(type.getContext(), 64);
      });

  // Setup the set of the patterns rewriter. At this point we want to
  // convert the `scf` operations to `std` and `std` operations to `llvm`.
  mlir::RewritePatternSet patterns(&getContext());
  mlir::concretelang::populateRTToLLVMConversionPatterns(typeConverter,
                                                         patterns);
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);

  // Apply a `FullConversion` to `llvm`.
  auto module = getOperation();
  if (mlir::applyFullConversion(module, target, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

llvm::Optional<mlir::Type>
MLIRLowerableDialectsToLLVMPass::convertTypes(mlir::Type type) {
  if (type.isa<mlir::concretelang::Concrete::LweCiphertextType>() ||
      type.isa<mlir::concretelang::Concrete::GlweCiphertextType>() ||
      type.isa<mlir::concretelang::Concrete::LweKeySwitchKeyType>() ||
      type.isa<mlir::concretelang::Concrete::LweBootstrapKeyType>() ||
      type.isa<mlir::concretelang::Concrete::ContextType>() ||
      type.isa<mlir::concretelang::Concrete::ForeignPlaintextListType>() ||
      type.isa<mlir::concretelang::Concrete::PlaintextListType>() ||
      type.isa<mlir::concretelang::RT::FutureType>()) {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(type.getContext(), 64));
  }
  if (type.isa<mlir::concretelang::RT::PointerType>()) {
    mlir::LowerToLLVMOptions options(type.getContext());
    mlir::LLVMTypeConverter typeConverter(type.getContext(), options);
    typeConverter.addConversion(convertTypes);
    typeConverter.addConversion(
        [&](mlir::concretelang::Concrete::PlaintextType type) {
          return mlir::IntegerType::get(type.getContext(), 64);
        });
    typeConverter.addConversion(
        [&](mlir::concretelang::Concrete::CleartextType type) {
          return mlir::IntegerType::get(type.getContext(), 64);
        });
    mlir::Type subtype =
        type.dyn_cast<mlir::concretelang::RT::PointerType>().getElementType();
    mlir::Type convertedSubtype = typeConverter.convertType(subtype);
    return mlir::LLVM::LLVMPointerType::get(convertedSubtype);
  }
  return llvm::None;
}

namespace mlir {
namespace concretelang {
/// Create a pass for lowering operations the remaining mlir dialects
/// operations, to the LLVM dialect for codegen.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMLIRLowerableDialectsToLLVMPass() {
  return std::make_unique<MLIRLowerableDialectsToLLVMPass>();
}
} // namespace concretelang
} // namespace mlir
