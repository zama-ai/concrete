// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Operation.h>

#include "concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h"
#include "concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/FHEToTFHECrt/Pass.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/Dialects/SCF.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/RT/IR/RTTypes.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace FHE = mlir::concretelang::FHE;
namespace TFHE = mlir::concretelang::TFHE;
namespace concretelang = mlir::concretelang;

namespace fhe_to_tfhe_crt_conversion {

namespace typing {

/// Converts `FHE::EncryptedInteger` into `Tensor<TFHE::GlweCiphetext>`.
mlir::RankedTensorType convertEint(mlir::MLIRContext *context,
                                   FHE::EncryptedIntegerType eint,
                                   uint64_t crtLength) {
  return mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>((int64_t)crtLength),
      TFHE::GLWECipherTextType::get(context, -1, -1, -1, eint.getWidth()));
}

/// Converts `Tensor<FHE::EncryptedInteger>` into a
/// `Tensor<TFHE::GlweCiphertext>` if the element type is appropriate. Otherwise
/// return the input type.
mlir::Type maybeConvertEintTensor(mlir::MLIRContext *context,
                                  mlir::RankedTensorType maybeEintTensor,
                                  uint64_t crtLength) {
  if (!maybeEintTensor.getElementType().isa<FHE::EncryptedIntegerType>()) {
    return (mlir::Type)(maybeEintTensor);
  }
  auto eint =
      maybeEintTensor.getElementType().cast<FHE::EncryptedIntegerType>();
  auto currentShape = maybeEintTensor.getShape();
  mlir::SmallVector<int64_t> newShape =
      mlir::SmallVector<int64_t>(currentShape.begin(), currentShape.end());
  newShape.push_back((int64_t)crtLength);
  return mlir::RankedTensorType::get(
      llvm::ArrayRef<int64_t>(newShape),
      TFHE::GLWECipherTextType::get(context, -1, -1, -1, eint.getWidth()));
}

/// Converts the type `FHE::EncryptedInteger` to `Tensor<TFHE::GlweCiphetext>`
/// if the input type is appropriate. Otherwise return the input type.
mlir::Type maybeConvertEint(mlir::MLIRContext *context, mlir::Type t,
                            uint64_t crtLength) {
  if (auto eint = t.dyn_cast<FHE::EncryptedIntegerType>())
    return convertEint(context, eint, crtLength);

  return t;
}

/// The type converter used to convert `FHE` to `TFHE` types using the crt
/// strategy.
class TypeConverter : public mlir::TypeConverter {

public:
  TypeConverter(concretelang::CrtLoweringParameters loweringParameters) {
    size_t nMods = loweringParameters.nMods;
    addConversion([](mlir::Type type) { return type; });
    addConversion([=](FHE::EncryptedIntegerType type) {
      return convertEint(type.getContext(), type, nMods);
    });
    addConversion([=](mlir::RankedTensorType type) {
      return maybeConvertEintTensor(type.getContext(), type, nMods);
    });
    addConversion([&](concretelang::RT::FutureType type) {
      return concretelang::RT::FutureType::get(this->convertType(
          type.dyn_cast<concretelang::RT::FutureType>().getElementType()));
    });
    addConversion([&](concretelang::RT::PointerType type) {
      return concretelang::RT::PointerType::get(this->convertType(
          type.dyn_cast<concretelang::RT::PointerType>().getElementType()));
    });
  }

  /// Returns a lambda that uses this converter to turn one type into another.
  std::function<mlir::Type(mlir::MLIRContext *, mlir::Type)>
  getConversionLambda() {
    return [&](mlir::MLIRContext *, mlir::Type t) { return convertType(t); };
  }
};

} // namespace typing

namespace lowering {

/// A pattern rewriter superclass used by most op rewriters during the
/// conversion.
template <typename T>
struct CrtOpPattern : public mlir::OpConversionPattern<T> {

  /// The lowering parameters are bound to the op rewriter.
  concretelang::CrtLoweringParameters loweringParameters;

  CrtOpPattern(mlir::MLIRContext *context,
               concretelang::CrtLoweringParameters params,
               mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<T>(typeConverter, context, benefit),
        loweringParameters(params), typeConverter(params) {}

  /// Writes an `scf::for` that loops over the crt dimension of one tensor and
  /// execute the input lambda to write the loop body. Returns the first result
  /// of the op.
  ///
  /// Note:
  /// -----
  ///
  /// + The type of `firstArgTensor` type is used as output type.
  mlir::Value writeUnaryTensorLoop(
      mlir::Location location, mlir::Type returnType,
      mlir::PatternRewriter &rewriter,
      mlir::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::Value,
                              mlir::ValueRange)>
          body) const {

    mlir::Value tensor = rewriter.create<mlir::bufferization::AllocTensorOp>(
        location, returnType.cast<mlir::RankedTensorType>(),
        mlir::ValueRange{});

    // Create the loop
    mlir::arith::ConstantOp zeroConstantOp =
        rewriter.create<mlir::arith::ConstantIndexOp>(location, 0);
    mlir::arith::ConstantOp oneConstantOp =
        rewriter.create<mlir::arith::ConstantIndexOp>(location, 1);
    mlir::arith::ConstantOp crtSizeConstantOp =
        rewriter.create<mlir::arith::ConstantIndexOp>(location,
                                                      loweringParameters.nMods);
    mlir::scf::ForOp newOp = rewriter.create<mlir::scf::ForOp>(
        location, zeroConstantOp, crtSizeConstantOp, oneConstantOp, tensor,
        body);

    return newOp.getResult(0);
  }

  /// Writes the crt encoding of a plaintext of arbitrary precision.
  mlir::Value writePlaintextCrtEncoding(mlir::Location location,
                                        mlir::Value rawPlaintext,
                                        mlir::PatternRewriter &rewriter) const {
    mlir::Value castedPlaintext = rewriter.create<mlir::arith::ExtSIOp>(
        location, rewriter.getI64Type(), rawPlaintext);
    return rewriter.create<TFHE::EncodePlaintextWithCrtOp>(
        location,
        mlir::RankedTensorType::get(
            mlir::ArrayRef<int64_t>(loweringParameters.nMods),
            rewriter.getI64Type()),
        castedPlaintext, rewriter.getI64ArrayAttr(loweringParameters.mods),
        rewriter.getI64IntegerAttr(loweringParameters.modsProd));
  }

protected:
  typing::TypeConverter typeConverter;
};

/// Rewriter for the `FHE::add_eint_int` operation.
struct AddEintIntOpPattern : public CrtOpPattern<FHE::AddEintIntOp> {

  AddEintIntOpPattern(mlir::MLIRContext *context,
                      concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::AddEintIntOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::AddEintIntOp op, FHE::AddEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.a();
    mlir::Value intOperand = adaptor.b();

    // Write plaintext encoding
    mlir::Value encodedPlaintextTensor =
        writePlaintextCrtEncoding(op.getLoc(), intOperand, rewriter);

    // Write add loop.
    mlir::Type ciphertextScalarType =
        converter->convertType(eintOperand.getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();
    mlir::Value output = writeUnaryTensorLoop(
        location, eintOperand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedEint =
              builder.create<mlir::tensor::ExtractOp>(loc, eintOperand, iter);
          mlir::Value extractedInt = builder.create<mlir::tensor::ExtractOp>(
              loc, encodedPlaintextTensor, iter);
          mlir::Value output = builder.create<TFHE::AddGLWEIntOp>(
              loc, ciphertextScalarType, extractedEint, extractedInt);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, output, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::sub_int_eint` operation.
struct SubIntEintOpPattern : public CrtOpPattern<FHE::SubIntEintOp> {

  SubIntEintOpPattern(mlir::MLIRContext *context,
                      concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::SubIntEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubIntEintOp op, FHE::SubIntEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value intOperand = adaptor.a();
    mlir::Value eintOperand = adaptor.b();

    // Write plaintext encoding
    mlir::Value encodedPlaintextTensor =
        writePlaintextCrtEncoding(op.getLoc(), intOperand, rewriter);

    // Write add loop.
    mlir::Type ciphertextScalarType =
        converter->convertType(eintOperand.getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();
    mlir::Value output = writeUnaryTensorLoop(
        location, eintOperand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedEint =
              builder.create<mlir::tensor::ExtractOp>(loc, eintOperand, iter);
          mlir::Value extractedInt = builder.create<mlir::tensor::ExtractOp>(
              loc, encodedPlaintextTensor, iter);
          mlir::Value output = builder.create<TFHE::SubGLWEIntOp>(
              loc, ciphertextScalarType, extractedInt, extractedEint);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, output, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::sub_eint_int` operation.
struct SubEintIntOpPattern : public CrtOpPattern<FHE::SubEintIntOp> {

  SubEintIntOpPattern(mlir::MLIRContext *context,
                      concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::SubEintIntOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubEintIntOp op, FHE::SubEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.a();
    mlir::Value intOperand = adaptor.b();

    // Write plaintext negation
    mlir::Type intType = intOperand.getType();
    mlir::Attribute minusOneAttr = mlir::IntegerAttr::get(intType, -1);
    mlir::Value minusOne =
        rewriter.create<mlir::arith::ConstantOp>(location, minusOneAttr)
            .getResult();
    mlir::Value negative =
        rewriter.create<mlir::arith::MulIOp>(location, intOperand, minusOne)
            .getResult();

    // Write plaintext encoding
    mlir::Value encodedPlaintextTensor =
        writePlaintextCrtEncoding(op.getLoc(), negative, rewriter);

    // Write add loop.
    mlir::Type ciphertextScalarType =
        converter->convertType(eintOperand.getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();
    mlir::Value output = writeUnaryTensorLoop(
        location, eintOperand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedEint =
              builder.create<mlir::tensor::ExtractOp>(loc, eintOperand, iter);
          mlir::Value extractedInt = builder.create<mlir::tensor::ExtractOp>(
              loc, encodedPlaintextTensor, iter);
          mlir::Value output = builder.create<TFHE::AddGLWEIntOp>(
              loc, ciphertextScalarType, extractedEint, extractedInt);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, output, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::add_eint` operation.
struct AddEintOpPattern : CrtOpPattern<FHE::AddEintOp> {

  AddEintOpPattern(mlir::MLIRContext *context,
                   concretelang::CrtLoweringParameters params,
                   mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::AddEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::AddEintOp op, FHE::AddEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value lhsOperand = adaptor.a();
    mlir::Value rhsOperand = adaptor.b();

    // Write add loop.
    mlir::Type ciphertextScalarType =
        converter->convertType(lhsOperand.getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();
    mlir::Value output = writeUnaryTensorLoop(
        location, lhsOperand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedLhs =
              builder.create<mlir::tensor::ExtractOp>(loc, lhsOperand, iter);
          mlir::Value extractedRhs =
              builder.create<mlir::tensor::ExtractOp>(loc, rhsOperand, iter);
          mlir::Value output = builder.create<TFHE::AddGLWEOp>(
              loc, ciphertextScalarType, extractedLhs, extractedRhs);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, output, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::sub_eint` operation.
struct SubEintOpPattern : CrtOpPattern<FHE::SubEintOp> {

  SubEintOpPattern(mlir::MLIRContext *context,
                   concretelang::CrtLoweringParameters params,
                   mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::SubEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubEintOp op, FHE::SubEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value lhsOperand = adaptor.a();
    mlir::Value rhsOperand = adaptor.b();

    // Write sub loop.
    mlir::Type ciphertextScalarType =
        converter->convertType(lhsOperand.getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();
    mlir::Value output = writeUnaryTensorLoop(
        location, lhsOperand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedLhs =
              builder.create<mlir::tensor::ExtractOp>(loc, lhsOperand, iter);
          mlir::Value extractedRhs =
              builder.create<mlir::tensor::ExtractOp>(loc, rhsOperand, iter);
          mlir::Value negatedRhs = builder.create<TFHE::NegGLWEOp>(
              loc, ciphertextScalarType, extractedRhs);
          mlir::Value output = builder.create<TFHE::AddGLWEOp>(
              loc, ciphertextScalarType, extractedLhs, negatedRhs);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, output, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::neg_eint` operation.
struct NegEintOpPattern : CrtOpPattern<FHE::NegEintOp> {

  NegEintOpPattern(mlir::MLIRContext *context,
                   concretelang::CrtLoweringParameters params,
                   mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::NegEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::NegEintOp op, FHE::NegEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value operand = adaptor.a();

    // Write the loop nest.
    mlir::Type ciphertextScalarType = converter->convertType(operand.getType())
                                          .cast<mlir::RankedTensorType>()
                                          .getElementType();
    mlir::Value loopRes = writeUnaryTensorLoop(
        location, operand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedCiphertext =
              builder.create<mlir::tensor::ExtractOp>(loc, operand, iter);
          mlir::Value negatedCiphertext = builder.create<TFHE::NegGLWEOp>(
              loc, ciphertextScalarType, extractedCiphertext);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, negatedCiphertext, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, loopRes);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::mul_eint_int` operation.
struct MulEintIntOpPattern : CrtOpPattern<FHE::MulEintIntOp> {

  MulEintIntOpPattern(mlir::MLIRContext *context,
                      concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::MulEintIntOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::MulEintIntOp op, FHE::MulEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.a();
    mlir::Value intOperand = adaptor.b();

    // Write cleartext "encoding"
    mlir::Value encodedCleartext = rewriter.create<mlir::arith::ExtSIOp>(
        location, rewriter.getI64Type(), intOperand);

    // Write the loop nest.
    mlir::Type ciphertextScalarType =
        converter->convertType(eintOperand.getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();
    mlir::Value loopRes = writeUnaryTensorLoop(
        location, eintOperand.getType(), rewriter,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          mlir::Value extractedCiphertext =
              builder.create<mlir::tensor::ExtractOp>(loc, eintOperand, iter);
          mlir::Value negatedCiphertext = builder.create<TFHE::MulGLWEIntOp>(
              loc, ciphertextScalarType, extractedCiphertext, encodedCleartext);
          mlir::Value newTensor = builder.create<mlir::tensor::InsertOp>(
              loc, negatedCiphertext, args[0], iter);
          builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newTensor});
        });

    // Rewrite original op.
    rewriter.replaceOp(op, loopRes);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::apply_lookup_table` operation.
struct ApplyLookupTableEintOpPattern
    : public CrtOpPattern<FHE::ApplyLookupTableEintOp> {

  ApplyLookupTableEintOpPattern(mlir::MLIRContext *context,
                                concretelang::CrtLoweringParameters params,
                                mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::ApplyLookupTableEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::ApplyLookupTableEintOp op,
                  FHE::ApplyLookupTableEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();

    mlir::Value newLut =
        rewriter
            .create<TFHE::EncodeExpandLutForWopPBSOp>(
                op.getLoc(),
                mlir::RankedTensorType::get(
                    mlir::ArrayRef<int64_t>(loweringParameters.lutSize),
                    rewriter.getI64Type()),
                adaptor.lut(),
                rewriter.getI64ArrayAttr(
                    mlir::ArrayRef<int64_t>(loweringParameters.mods)),
                rewriter.getI64ArrayAttr(
                    mlir::ArrayRef<int64_t>(loweringParameters.bits)),
                rewriter.getI32IntegerAttr(loweringParameters.polynomialSize),
                rewriter.getI32IntegerAttr(loweringParameters.modsProd))
            .getResult();

    // Replace the lut with an encoded / expanded one.
    auto wopPBS = rewriter.create<TFHE::WopPBSGLWEOp>(
        op.getLoc(), converter->convertType(op.getType()), adaptor.a(), newLut,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, rewriter.getI64ArrayAttr({}));

    rewriter.replaceOp(op, {wopPBS.getResult()});
    return ::mlir::success();
  };
};

/// Rewriter for the `tensor::extract` operation.
struct TensorExtractOpPattern : public CrtOpPattern<mlir::tensor::ExtractOp> {

  TensorExtractOpPattern(mlir::MLIRContext *context,
                         concretelang::CrtLoweringParameters params,
                         mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<mlir::tensor::ExtractOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractOp op,
                  mlir::tensor::ExtractOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();

    if (!op.getTensor()
             .getType()
             .cast<mlir::TensorType>()
             .getElementType()
             .isa<FHE::EncryptedIntegerType>() &&
        !op.getTensor()
             .getType()
             .cast<mlir::TensorType>()
             .getElementType()
             .isa<TFHE::GLWECipherTextType>()) {
      return mlir::success();
    }

    mlir::SmallVector<mlir::OpFoldResult> offsets;
    mlir::SmallVector<mlir::OpFoldResult> sizes;
    mlir::SmallVector<mlir::OpFoldResult> strides;
    for (auto index : op.getIndices()) {
      offsets.push_back(index);
      sizes.push_back(rewriter.getI64IntegerAttr(1));
      strides.push_back(rewriter.getI64IntegerAttr(1));
    }
    offsets.push_back(
        rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0)
            .getResult());
    sizes.push_back(rewriter.getI64IntegerAttr(loweringParameters.nMods));
    strides.push_back(rewriter.getI64IntegerAttr(1));
    auto newOp = rewriter.create<mlir::tensor::ExtractSliceOp>(
        op.getLoc(),
        converter->convertType(op.getResult().getType())
            .cast<mlir::RankedTensorType>(),
        adaptor.getTensor(), offsets, sizes, strides);

    rewriter.replaceOp(op, {newOp.getResult()});
    return mlir::success();
  }
};

/// Rewriter for the `tensor::extract` operation.
struct TensorInsertOpPattern : public CrtOpPattern<mlir::tensor::InsertOp> {

  TensorInsertOpPattern(mlir::MLIRContext *context,
                        concretelang::CrtLoweringParameters params,
                        mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<mlir::tensor::InsertOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertOp op,
                  mlir::tensor::InsertOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getDest()
             .getType()
             .cast<mlir::TensorType>()
             .getElementType()
             .isa<FHE::EncryptedIntegerType>() &&
        !op.getDest()
             .getType()
             .cast<mlir::TensorType>()
             .getElementType()
             .isa<TFHE::GLWECipherTextType>()) {
      return mlir::success();
    }

    mlir::SmallVector<mlir::OpFoldResult> offsets;
    mlir::SmallVector<mlir::OpFoldResult> sizes;
    mlir::SmallVector<mlir::OpFoldResult> strides;
    for (auto index : op.getIndices()) {
      offsets.push_back(index);
      sizes.push_back(rewriter.getI64IntegerAttr(1));
      strides.push_back(rewriter.getI64IntegerAttr(1));
    }
    offsets.push_back(
        rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0)
            .getResult());
    sizes.push_back(rewriter.getI64IntegerAttr(loweringParameters.nMods));
    strides.push_back(rewriter.getI64IntegerAttr(1));
    auto newOp = rewriter.create<mlir::tensor::InsertSliceOp>(
        op.getLoc(), adaptor.getScalar(), op.getDest(), offsets, sizes,
        strides);

    rewriter.replaceOp(op, {newOp});
    return mlir::success();
  }
};

/// Rewriter for the `tensor::from_elements` operation.
struct TensorFromElementsOpPattern
    : public CrtOpPattern<mlir::tensor::FromElementsOp> {

  TensorFromElementsOpPattern(mlir::MLIRContext *context,
                              concretelang::CrtLoweringParameters params,
                              mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<mlir::tensor::FromElementsOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::FromElementsOp op,
                  mlir::tensor::FromElementsOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();

    if (!op.getResult()
             .getType()
             .cast<mlir::RankedTensorType>()
             .getElementType()
             .isa<FHE::EncryptedIntegerType>() &&
        !op.getResult()
             .getType()
             .cast<mlir::RankedTensorType>()
             .getElementType()
             .isa<TFHE::GLWECipherTextType>()) {
      return mlir::success();
    }

    // Create dest tensor allocation op
    mlir::Value outputTensor =
        rewriter.create<mlir::bufferization::AllocTensorOp>(
            op.getLoc(),
            converter->convertType(op.getResult().getType())
                .cast<mlir::RankedTensorType>(),
            mlir::ValueRange{});

    // Create insert_slice ops to insert the different pieces.
    auto outputShape =
        outputTensor.getType().cast<mlir::RankedTensorType>().getShape();
    mlir::SmallVector<mlir::OpFoldResult> offsets{
        rewriter.getI64IntegerAttr(0)};
    mlir::SmallVector<mlir::OpFoldResult> sizes{rewriter.getI64IntegerAttr(1)};

    mlir::SmallVector<mlir::OpFoldResult> strides{
        rewriter.getI64IntegerAttr(1)};
    for (size_t dimIndex = 1; dimIndex < outputShape.size(); ++dimIndex) {
      sizes.push_back(rewriter.getI64IntegerAttr(outputShape[dimIndex]));
      strides.push_back(rewriter.getI64IntegerAttr(1));
      offsets.push_back(rewriter.getI64IntegerAttr(0));
    }
    for (size_t insertionIndex = 0;
         insertionIndex < adaptor.getElements().size(); ++insertionIndex) {
      offsets[0] = rewriter.getI64IntegerAttr(insertionIndex);
      mlir::tensor::InsertSliceOp insertOp =
          rewriter.create<mlir::tensor::InsertSliceOp>(
              op.getLoc(), adaptor.getElements()[insertionIndex], outputTensor,
              offsets, sizes, strides);
      outputTensor = insertOp.getResult();
    }
    rewriter.replaceOp(op, {outputTensor});
    return mlir::success();
  }
};

} // namespace lowering

struct FHEToTFHECrtPass : public FHEToTFHECrtBase<FHEToTFHECrtPass> {

  FHEToTFHECrtPass(concretelang::CrtLoweringParameters params)
      : loweringParameters(params) {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    typing::TypeConverter converter(loweringParameters);

    //------------------------------------------- Marking legal/illegal dialects
    target.addIllegalDialect<FHE::FHEDialect>();
    target.addLegalDialect<TFHE::TFHEDialect>();
    target.addLegalDialect<mlir::arith::ArithmeticDialect>();
    target.addDynamicallyLegalOp<mlir::tensor::GenerateOp, mlir::scf::ForOp>(
        [&](mlir::Operation *op) {
          return (
              converter.isLegal(op->getOperandTypes()) &&
              converter.isLegal(op->getResultTypes()) &&
              converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
        });
    target.addDynamicallyLegalOp<mlir::tensor::InsertOp,
                                 mlir::tensor::ExtractOp, mlir::scf::YieldOp>(
        [&](mlir::Operation *op) {
          return (converter.isLegal(op->getOperandTypes()) &&
                  converter.isLegal(op->getResultTypes()));
        });
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return converter.isSignatureLegal(funcOp.getFunctionType()) &&
                 converter.isLegal(&funcOp.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
        [&](mlir::func::ConstantOp op) {
          return FunctionConstantOpConversion<typing::TypeConverter>::isLegal(
              op, converter);
        });
    target.addLegalOp<mlir::func::CallOp>();

    concretelang::addDynamicallyLegalTypeOp<mlir::bufferization::AllocTensorOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(target,
                                                                  converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::tensor::ExtractSliceOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::tensor::InsertSliceOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::tensor::FromElementsOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::tensor::ExpandShapeOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::tensor::CollapseShapeOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::MakeReadyFutureOp>(target, converter);
    concretelang::addDynamicallyLegalTypeOp<concretelang::RT::AwaitFutureOp>(
        target, converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::CreateAsyncTaskOp>(target, converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::BuildReturnPtrPlaceholderOp>(target, converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(target,
                                                                     converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::DerefReturnPtrPlaceholderOp>(target, converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::WorkFunctionReturnOp>(target, converter);
    concretelang::addDynamicallyLegalTypeOp<
        concretelang::RT::RegisterTaskWorkFunctionOp>(target, converter);

    //---------------------------------------------------------- Adding patterns
    mlir::RewritePatternSet patterns(&getContext());

    // Patterns for `bufferization` dialect operations.
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::bufferization::AllocTensorOp, true>>(patterns.getContext(),
                                                   converter);

    // Patterns for the `FHE` dialect operations
    patterns.add<
        //    |_ `FHE::zero_eint`
        concretelang::GenericOneToOneOpConversionPattern<FHE::ZeroEintOp,
                                                         TFHE::ZeroGLWEOp>,
        //    |_ `FHE::zero_tensor`
        concretelang::GenericOneToOneOpConversionPattern<
            FHE::ZeroTensorOp, TFHE::ZeroTensorGLWEOp>>(&getContext(),
                                                        converter);
    //    |_ `FHE::add_eint_int`
    patterns.add<lowering::AddEintIntOpPattern,
                 //    |_ `FHE::add_eint`
                 lowering::AddEintOpPattern,
                 //    |_ `FHE::sub_int_eint`
                 lowering::SubIntEintOpPattern,
                 //    |_ `FHE::sub_eint_int`
                 lowering::SubEintIntOpPattern,
                 //    |_ `FHE::sub_eint`
                 lowering::SubEintOpPattern,
                 //    |_ `FHE::neg_eint`
                 lowering::NegEintOpPattern,
                 //    |_ `FHE::mul_eint_int`
                 lowering::MulEintIntOpPattern,
                 //    |_ `FHE::apply_lookup_table`
                 lowering::ApplyLookupTableEintOpPattern>(&getContext(),
                                                          loweringParameters);

    // Patterns for the relics of the `FHELinalg` dialect operations.
    //    |_ `linalg::generic` turned to nested `scf::for`
    patterns.add<
        concretelang::TypeConvertingReinstantiationPattern<mlir::scf::ForOp>>(
        patterns.getContext(), converter);
    patterns.add<
        concretelang::TypeConvertingReinstantiationPattern<mlir::scf::YieldOp>>(
        patterns.getContext(), converter);
    patterns.add<
        concretelang::TypeConvertingReinstantiationPattern<mlir::scf::ForOp>>(
        &getContext(), converter);
    patterns.add<lowering::TensorExtractOpPattern>(&getContext(),
                                                   loweringParameters);
    patterns.add<lowering::TensorInsertOpPattern>(&getContext(),
                                                  loweringParameters);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::tensor::ExtractSliceOp>>(patterns.getContext(), converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::tensor::InsertSliceOp>>(patterns.getContext(), converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::tensor::CollapseShapeOp>>(patterns.getContext(), converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::tensor::ExpandShapeOp>>(patterns.getContext(), converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::tensor::GenerateOp, true>>(&getContext(), converter);

    // Patterns for `func` dialect operations.
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::func::ReturnOp>>(patterns.getContext(), converter);
    patterns.add<FunctionConstantOpConversion<typing::TypeConverter>>(
        &getContext(), converter);

    // Pattern for the `tensor::from_element` op.
    patterns.add<lowering::TensorFromElementsOpPattern>(patterns.getContext(),
                                                        loweringParameters);

    // Patterns for the `RT` dialect operations.
    patterns.add<
        // concretelang::TypeConvertingReinstantiationPattern<
        //     mlir::func::ReturnOp>,
        concretelang::TypeConvertingReinstantiationPattern<mlir::scf::YieldOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::MakeReadyFutureOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::AwaitFutureOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::CreateAsyncTaskOp, true>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::BuildReturnPtrPlaceholderOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::DerefReturnPtrPlaceholderOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::WorkFunctionReturnOp>,
        concretelang::TypeConvertingReinstantiationPattern<
            concretelang::RT::RegisterTaskWorkFunctionOp>>(&getContext(),
                                                           converter);

    //--------------------------------------------------------- Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }

private:
  concretelang::CrtLoweringParameters loweringParameters;
};
} // namespace fhe_to_tfhe_crt_conversion

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertFHEToTFHECrtPass(CrtLoweringParameters lowering) {
  return std::make_unique<fhe_to_tfhe_crt_conversion::FHEToTFHECrtPass>(
      lowering);
}
} // namespace concretelang
} // namespace mlir
