// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/RT/Analysis/Autopar.h"
#include "concretelang/Dialect/RT/IR/RTTypes.h"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"

namespace {
struct MLIRLowerableDialectsToLLVMPass
    : public MLIRLowerableDialectsToLLVMBase<MLIRLowerableDialectsToLLVMPass> {
  void runOnOperation() final;

  /// Convert types to the LLVM dialect-compatible type
  static std::optional<mlir::Type> convertTypes(mlir::Type type);
};
} // namespace

/// This rewrite pattern transforms any instance of `memref.copy`
/// operators on 1D memref.
/// This is introduced to avoid the MLIR lowering of `memref.copy` of ranked
/// memref that basically allocate unranked memref structure on the stack before
/// calling @memrefCopy.
///
/// Example:
///
/// ```mlir
/// memref.copy %src, %dst : memref<Xxi64> to memref<Xxi64>
/// ```
///
/// becomes:
///
/// ```mlir
/// %_src = memref.cast %src = memref<Xxi64> to memref<?xi64>
/// %_dst = memref.cast %dst = memref<Xxi64> to memref<?xi64>
/// call @memref_copy_one_rank(%_src, %_dst) : (tensor<?xi64>, tensor<?xi64>) ->
/// ()
/// ```
struct Memref1DCopyOpPattern
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  Memref1DCopyOpPattern(mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::memref::CopyOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp copyOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (copyOp.getSource().getType().cast<mlir::MemRefType>().getRank() != 1 ||
        copyOp.getSource().getType().cast<mlir::MemRefType>().getRank() != 1) {
      return mlir::failure();
    }
    auto opType = mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                        rewriter.getI64Type());
    // Insert forward declaration of the add_lwe_ciphertexts function
    {
      if (insertForwardDeclaration(
              copyOp, rewriter, "memref_copy_one_rank",
              mlir::FunctionType::get(rewriter.getContext(), {opType, opType},
                                      {}))
              .failed()) {
        return mlir::failure();
      }
    }
    auto sourceOp = rewriter.create<mlir::memref::CastOp>(
        copyOp.getLoc(), opType, copyOp.getSource());
    auto targetOp = rewriter.create<mlir::memref::CastOp>(
        copyOp.getLoc(), opType, copyOp.getTarget());
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        copyOp, "memref_copy_one_rank", mlir::TypeRange{},
        mlir::ValueRange{sourceOp, targetOp});
    return mlir::success();
  };
};

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

  // Setup the set of the patterns rewriter. At this point we want to
  // convert the `scf` operations to `std` and `std` operations to `llvm`.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<Memref1DCopyOpPattern>(&getContext(), 100);
  mlir::concretelang::populateRTToLLVMConversionPatterns(typeConverter,
                                                         patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::memref::populateExpandStridedMetadataPatterns(patterns);
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  target.addLegalOp<mlir::scf::YieldOp>();
  mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

  target.addDynamicallyLegalOp<mlir::omp::MasterOp, mlir::omp::ParallelOp,
                               mlir::omp::WsLoopOp>([&](mlir::Operation *op) {
    return typeConverter.isLegal(&op->getRegion(0));
  });
  target.addLegalOp<mlir::omp::TerminatorOp, mlir::omp::TaskyieldOp,
                    mlir::omp::FlushOp, mlir::omp::BarrierOp,
                    mlir::omp::TaskwaitOp, mlir::omp::YieldOp>();

  // Apply a `FullConversion` to `llvm`.
  auto module = getOperation();
  if (mlir::applyFullConversion(module, target, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

std::optional<mlir::Type>
MLIRLowerableDialectsToLLVMPass::convertTypes(mlir::Type type) {
  if (type.isa<mlir::concretelang::Concrete::ContextType>() ||
      type.isa<mlir::concretelang::RT::FutureType>() ||
      type.isa<mlir::concretelang::SDFG::DFGType>() ||
      type.isa<mlir::concretelang::SDFG::StreamType>()) {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(type.getContext(), 64));
  }
  if (type.isa<mlir::concretelang::RT::PointerType>()) {
    mlir::LowerToLLVMOptions options(type.getContext());
    mlir::LLVMTypeConverter typeConverter(type.getContext(), options);
    typeConverter.addConversion(convertTypes);
    mlir::Type subtype =
        type.dyn_cast<mlir::concretelang::RT::PointerType>().getElementType();
    mlir::Type convertedSubtype = typeConverter.convertType(subtype);
    return mlir::LLVM::LLVMPointerType::get(convertedSubtype);
  }
  return std::nullopt;
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
