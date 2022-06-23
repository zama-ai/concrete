// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Operation.h>

#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/FHEToTFHE/Patterns.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace FHE = mlir::concretelang::FHE;
namespace TFHE = mlir::concretelang::TFHE;

namespace {
struct FHEToTFHEPass : public FHEToTFHEBase<FHEToTFHEPass> {
  void runOnOperation() final;
};
} // namespace

using mlir::concretelang::FHE::EncryptedIntegerType;
using mlir::concretelang::TFHE::GLWECipherTextType;

/// FHEToTFHETypeConverter is a TypeConverter that transform
/// `FHE.eint<p>` to `TFHE.glwe<{_,_,_}{p}>`
class FHEToTFHETypeConverter : public mlir::TypeConverter {

public:
  FHEToTFHETypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([](EncryptedIntegerType type) {
      return mlir::concretelang::convertTypeEncryptedIntegerToGLWE(
          type.getContext(), type);
    });
    addConversion([](mlir::RankedTensorType type) {
      auto eint =
          type.getElementType().dyn_cast_or_null<EncryptedIntegerType>();
      if (eint == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(),
          mlir::concretelang::convertTypeEncryptedIntegerToGLWE(
              eint.getContext(), eint));
      return r;
    });
  }
};

// This rewrite pattern transforms any instance of `FHE.apply_lookup_table`
// operators.
//
// Example:
//
// ```mlir
// %0 = "FHE.apply_lookup_table"(%ct, %lut): (!FHE.eint<2>, tensor<4xi64>)
//        ->(!FHE.eint<2>)
// ```
//
// becomes:
//
// ```mlir
//  %glwe_lut = "TFHE.glwe_from_table"(%lut)
//                : (tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{2}>
//  %glwe_ks = "TFHE.keyswitch_glwe"(%ct)
//               {baseLog = -1 : i32, level = -1 : i32}
//               : (!TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
//  %0 = "TFHE.bootstrap_glwe"(%glwe_ks, %glwe_lut)
//         {baseLog = -1 : i32, glweDimension = -1 : i32, level = -1 : i32,
//           polynomialSize = -1 : i32}
//         : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) ->
//         !TFHE.glwe<{_,_,_}{2}>
// ```
struct ApplyLookupTableEintOpPattern
    : public mlir::OpRewritePattern<FHE::ApplyLookupTableEintOp> {
  ApplyLookupTableEintOpPattern(mlir::MLIRContext *context,
                                mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<FHE::ApplyLookupTableEintOp>(context,
                                                              benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::ApplyLookupTableEintOp lutOp,
                  mlir::PatternRewriter &rewriter) const override {
    FHEToTFHETypeConverter converter;
    auto inputTy = converter.convertType(lutOp.a().getType())
                       .cast<TFHE::GLWECipherTextType>();
    auto resultTy = converter.convertType(lutOp.getType());
    //  %glwe_lut = "TFHE.glwe_from_table"(%lut)
    auto glweLut = rewriter.create<TFHE::GLWEFromTableOp>(lutOp.getLoc(),
                                                          inputTy, lutOp.lut());
    //  %glwe_ks = "TFHE.keyswitch_glwe"(%ct)
    auto glweKs = rewriter.create<TFHE::KeySwitchGLWEOp>(
        lutOp.getLoc(), inputTy, lutOp.a(), -1, -1);
    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, glweKs, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });
    //  %0 = "TFHE.bootstrap_glwe"(%glwe_ks, %glwe_lut)
    rewriter.replaceOpWithNewOp<TFHE::BootstrapGLWEOp>(lutOp, resultTy, glweKs,
                                                       glweLut, -1, -1);
    return ::mlir::success();
  };
};

// This rewrite pattern transforms any instance of `FHE.sub_eint_int`
// operators to a negation and an addition.
struct SubEintIntOpPattern : public mlir::OpRewritePattern<FHE::SubEintIntOp> {
  SubEintIntOpPattern(mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<FHE::SubEintIntOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubEintIntOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location location = op.getLoc();

    mlir::Value lhs = op.getOperand(0);
    mlir::Value rhs = op.getOperand(1);

    mlir::Type rhsType = rhs.getType();
    mlir::Attribute minusOneAttr = mlir::IntegerAttr::get(rhsType, -1);
    mlir::Value minusOne =
        rewriter.create<mlir::arith::ConstantOp>(location, minusOneAttr)
            .getResult();

    mlir::Value negative =
        rewriter.create<mlir::arith::MulIOp>(location, rhs, minusOne)
            .getResult();

    FHEToTFHETypeConverter converter;
    auto resultTy = converter.convertType(op.getType());

    auto addition =
        rewriter.create<TFHE::AddGLWEIntOp>(location, resultTy, lhs, negative);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, addition, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    rewriter.replaceOp(op, {addition.getResult()});
    return mlir::success();
  };
};

// This rewrite pattern transforms any instance of `FHE.sub_eint`
// operators to a negation and an addition.
struct SubEintOpPattern : public mlir::OpRewritePattern<FHE::SubEintOp> {
  SubEintOpPattern(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<FHE::SubEintOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubEintOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location location = op.getLoc();

    mlir::Value lhs = op.getOperand(0);
    mlir::Value rhs = op.getOperand(1);

    FHEToTFHETypeConverter converter;

    auto rhsTy = converter.convertType(rhs.getType());
    auto negative = rewriter.create<TFHE::NegGLWEOp>(location, rhsTy, rhs);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, negative, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    auto resultTy = converter.convertType(op.getType());
    auto addition = rewriter.create<TFHE::AddGLWEOp>(location, resultTy, lhs,
                                                     negative.getResult());

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, addition, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    rewriter.replaceOp(op, {addition.getResult()});
    return mlir::success();
  };
};

void FHEToTFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  FHEToTFHETypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::concretelang::TFHE::TFHEDialect>();
  target.addLegalDialect<mlir::arith::ArithmeticDialect>();

  // Make sure that no ops from `FHE` remain after the lowering
  target.addIllegalDialect<mlir::concretelang::FHE::FHEDialect>();

  // Make sure that no ops `linalg.generic` that have illegal types
  target
      .addDynamicallyLegalOp<mlir::linalg::GenericOp, mlir::tensor::GenerateOp>(
          [&](mlir::Operation *op) {
            return (
                converter.isLegal(op->getOperandTypes()) &&
                converter.isLegal(op->getResultTypes()) &&
                converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
          });

  // Make sure that func has legal signature
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp funcOp) {
        return converter.isSignatureLegal(funcOp.getFunctionType()) &&
               converter.isLegal(&funcOp.getBody());
      });

  // Add all patterns required to lower all ops from `FHE` to
  // `TFHE`
  mlir::RewritePatternSet patterns(&getContext());

  populateWithGeneratedFHEToTFHE(patterns);

  patterns.add<
      mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>>(
      patterns.getContext(), converter);

  patterns.add<ApplyLookupTableEintOpPattern>(&getContext());
  patterns.add<SubEintOpPattern>(&getContext());
  patterns.add<SubEintIntOpPattern>(&getContext());

  patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                            FHEToTFHETypeConverter>>(
      &getContext(), converter);

  patterns.add<
      mlir::concretelang::GenericTypeConverterPattern<mlir::linalg::YieldOp>>(
      patterns.getContext(), converter);

  patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                            FHEToTFHETypeConverter>>(
      &getContext(), converter);

  patterns.add<
      RegionOpTypeConverterPattern<mlir::scf::ForOp, FHEToTFHETypeConverter>>(
      &getContext(), converter);
  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      mlir::concretelang::FHE::ZeroTensorOp,
      mlir::concretelang::TFHE::ZeroTensorGLWEOp>>(&getContext(), converter);

  mlir::concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                              converter);

  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);

  // Conversion of RT Dialect Ops
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<
      mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(),
                                               converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowTaskOp>(target, converter);
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<
      mlir::concretelang::RT::DataflowYieldOp>>(patterns.getContext(),
                                                converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowYieldOp>(target, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertFHEToTFHEPass() {
  return std::make_unique<FHEToTFHEPass>();
}
} // namespace concretelang
} // namespace mlir
