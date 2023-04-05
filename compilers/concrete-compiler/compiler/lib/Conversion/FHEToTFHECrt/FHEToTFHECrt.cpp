// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>

#include "concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h"
#include "concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/FHEToTFHECrt/Pass.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/Dialects/SCF.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/RTOpConverter.h"
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
#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"

namespace FHE = mlir::concretelang::FHE;
namespace TFHE = mlir::concretelang::TFHE;
namespace Tracing = mlir::concretelang::Tracing;

namespace fhe_to_tfhe_crt_conversion {

namespace typing {

/// Converts an encrypted integer into `TFHE::GlweCiphertext`.
mlir::RankedTensorType convertEncrypted(mlir::MLIRContext *context,
                                        FHE::FheIntegerInterface enc,
                                        uint64_t crtLength) {
  return mlir::RankedTensorType::get(
      mlir::ArrayRef<int64_t>((int64_t)crtLength),
      TFHE::GLWECipherTextType::get(context, TFHE::GLWESecretKey()));
}

/// Converts `Tensor<FHE::AnyEncryptedInteger>` into a
/// `Tensor<TFHE::GlweCiphertext>` if the element type is appropriate.
/// Otherwise return the input type.
mlir::Type
maybeConvertEncryptedTensor(mlir::MLIRContext *context,
                            mlir::RankedTensorType maybeEncryptedTensor,
                            uint64_t crtLength) {
  if (!maybeEncryptedTensor.getElementType().isa<FHE::FheIntegerInterface>()) {
    return (mlir::Type)(maybeEncryptedTensor);
  }
  auto currentShape = maybeEncryptedTensor.getShape();
  mlir::SmallVector<int64_t> newShape =
      mlir::SmallVector<int64_t>(currentShape.begin(), currentShape.end());
  newShape.push_back((int64_t)crtLength);
  return mlir::RankedTensorType::get(
      llvm::ArrayRef<int64_t>(newShape),
      TFHE::GLWECipherTextType::get(context, TFHE::GLWESecretKey()));
}

/// Converts any encrypted type to `TFHE::GlweCiphetext` if the
/// input type is appropriate. Otherwise return the input type.
mlir::Type maybeConvertEncrypted(mlir::MLIRContext *context, mlir::Type t,
                                 uint64_t crtLength) {
  if (auto eint = t.dyn_cast<FHE::FheIntegerInterface>())
    return convertEncrypted(context, eint, crtLength);

  return t;
}

/// The type converter used to convert `FHE` to `TFHE` types using the crt
/// strategy.
class TypeConverter : public mlir::TypeConverter {

public:
  TypeConverter(mlir::concretelang::CrtLoweringParameters loweringParameters) {
    size_t nMods = loweringParameters.nMods;
    addConversion([](mlir::Type type) { return type; });
    addConversion([=](FHE::FheIntegerInterface type) {
      return convertEncrypted(type.getContext(), type, nMods);
    });
    addConversion([=](mlir::RankedTensorType type) {
      return maybeConvertEncryptedTensor(type.getContext(), type, nMods);
    });
    addConversion([&](mlir::concretelang::RT::FutureType type) {
      return mlir::concretelang::RT::FutureType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::FutureType>()
                                .getElementType()));
    });
    addConversion([&](mlir::concretelang::RT::PointerType type) {
      return mlir::concretelang::RT::PointerType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::PointerType>()
                                .getElementType()));
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
  mlir::concretelang::CrtLoweringParameters loweringParameters;

  CrtOpPattern(mlir::MLIRContext *context,
               mlir::concretelang::CrtLoweringParameters params,
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
                      mlir::concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::AddEintIntOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::AddEintIntOp op, FHE::AddEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.getA();
    mlir::Value intOperand = adaptor.getB();

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
                      mlir::concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::SubIntEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubIntEintOp op, FHE::SubIntEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value intOperand = adaptor.getA();
    mlir::Value eintOperand = adaptor.getB();

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
                      mlir::concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::SubEintIntOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubEintIntOp op, FHE::SubEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.getA();
    mlir::Value intOperand = adaptor.getB();

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
                   mlir::concretelang::CrtLoweringParameters params,
                   mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::AddEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::AddEintOp op, FHE::AddEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value lhsOperand = adaptor.getA();
    mlir::Value rhsOperand = adaptor.getB();

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
                   mlir::concretelang::CrtLoweringParameters params,
                   mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::SubEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::SubEintOp op, FHE::SubEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value lhsOperand = adaptor.getA();
    mlir::Value rhsOperand = adaptor.getB();

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
                   mlir::concretelang::CrtLoweringParameters params,
                   mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::NegEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::NegEintOp op, FHE::NegEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value operand = adaptor.getA();

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

/// Rewriter for the `FHE::to_signed` operation.
struct ToSignedOpPattern : public CrtOpPattern<FHE::ToSignedOp> {
  ToSignedOpPattern(mlir::MLIRContext *context,
                    mlir::concretelang::CrtLoweringParameters params,
                    mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::ToSignedOp>(context, params, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ToSignedOp op, FHE::ToSignedOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    typing::TypeConverter converter{loweringParameters};
    rewriter.replaceOp(op, {adaptor.getInput()});

    return mlir::success();
  }
};

/// Rewriter for the `FHE::to_unsigned` operation.
struct ToUnsignedOpPattern : public CrtOpPattern<FHE::ToUnsignedOp> {
  ToUnsignedOpPattern(mlir::MLIRContext *context,
                      mlir::concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::ToUnsignedOp>(context, params, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ToUnsignedOp op, FHE::ToUnsignedOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    typing::TypeConverter converter{loweringParameters};
    rewriter.replaceOp(op, {adaptor.getInput()});

    return mlir::success();
  }
};

/// Rewriter for the `FHE::mul_eint_int` operation.
struct MulEintIntOpPattern : CrtOpPattern<FHE::MulEintIntOp> {

  MulEintIntOpPattern(mlir::MLIRContext *context,
                      mlir::concretelang::CrtLoweringParameters params,
                      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::MulEintIntOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::MulEintIntOp op, FHE::MulEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.getA();
    mlir::Value intOperand = adaptor.getB();

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

  ApplyLookupTableEintOpPattern(
      mlir::MLIRContext *context,
      mlir::concretelang::CrtLoweringParameters params,
      mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::ApplyLookupTableEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::ApplyLookupTableEintOp op,
                  FHE::ApplyLookupTableEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter *converter = this->getTypeConverter();

    auto originalInputType =
        op.getA().getType().cast<FHE::FheIntegerInterface>();

    mlir::Value newLut =
        rewriter
            .create<TFHE::EncodeLutForCrtWopPBSOp>(
                op.getLoc(),
                mlir::RankedTensorType::get(
                    mlir::ArrayRef<int64_t>{
                        (int64_t)loweringParameters.nMods,
                        (int64_t)loweringParameters.singleLutSize},
                    rewriter.getI64Type()),
                adaptor.getLut(),
                rewriter.getI64ArrayAttr(
                    mlir::ArrayRef<int64_t>(loweringParameters.mods)),
                rewriter.getI64ArrayAttr(
                    mlir::ArrayRef<int64_t>(loweringParameters.bits)),
                rewriter.getI32IntegerAttr(loweringParameters.modsProd),
                rewriter.getBoolAttr(originalInputType.isSigned()))
            .getResult();

    // Replace the lut with an encoded / expanded one.
    auto wopPBS = rewriter.create<TFHE::WopPBSGLWEOp>(
        op.getLoc(), converter->convertType(op.getType()), adaptor.getA(),
        newLut,
        TFHE::GLWEKeyswitchKeyAttr::get(op.getContext(), TFHE::GLWESecretKey(),
                                        TFHE::GLWESecretKey(), -1, -1, -1),
        TFHE::GLWEBootstrapKeyAttr::get(op.getContext(), TFHE::GLWESecretKey(),
                                        TFHE::GLWESecretKey(), -1, -1, -1, -1,
                                        -1),
        TFHE::GLWEPackingKeyswitchKeyAttr::get(
            op.getContext(), TFHE::GLWESecretKey(), TFHE::GLWESecretKey(), -1,
            -1, -1, -1, -1, -1),
        rewriter.getI64ArrayAttr({}), rewriter.getI32IntegerAttr(-1),
        rewriter.getI32IntegerAttr(-1));

    rewriter.replaceOp(op, {wopPBS.getResult()});
    return ::mlir::success();
  };
};

/// Rewriter for the `Tracing::trace_ciphertext` operation.
struct TraceCiphertextOpPattern : CrtOpPattern<Tracing::TraceCiphertextOp> {

  TraceCiphertextOpPattern(mlir::MLIRContext *context,
                           mlir::concretelang::CrtLoweringParameters params,
                           mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<Tracing::TraceCiphertextOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(Tracing::TraceCiphertextOp op,
                  Tracing::TraceCiphertextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    typing::TypeConverter converter{loweringParameters};
    mlir::Type ciphertextScalarType =
        converter.convertType(op.getCiphertext().getType())
            .cast<mlir::RankedTensorType>()
            .getElementType();

    for (size_t i = 0; i < (loweringParameters.nMods - 1); ++i) {
      auto extractedCiphertext = rewriter.create<mlir::tensor::ExtractOp>(
          op.getLoc(), ciphertextScalarType, adaptor.getCiphertext(),
          mlir::ValueRange{rewriter.create<mlir::arith::ConstantOp>(
              op.getLoc(), rewriter.getIndexAttr(i))});
      rewriter.create<Tracing::TraceCiphertextOp>(
          op.getLoc(), extractedCiphertext, op.getMsgAttr(), op.getNmsbAttr());
    }

    auto extractedCiphertext = rewriter.create<mlir::tensor::ExtractOp>(
        op.getLoc(), ciphertextScalarType, adaptor.getCiphertext(),
        mlir::ValueRange{rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(loweringParameters.nMods - 1))});
    rewriter.replaceOpWithNewOp<Tracing::TraceCiphertextOp>(
        op, extractedCiphertext, op.getMsgAttr(), op.getNmsbAttr());

    return mlir::success();
  }
};

/// Rewriter for the `tensor::extract` operation.
struct TensorExtractOpPattern : public CrtOpPattern<mlir::tensor::ExtractOp> {

  TensorExtractOpPattern(mlir::MLIRContext *context,
                         mlir::concretelang::CrtLoweringParameters params,
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
             .isa<FHE::FheIntegerInterface>() &&
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
                        mlir::concretelang::CrtLoweringParameters params,
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
             .isa<FHE::FheIntegerInterface>() &&
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
        op.getLoc(), adaptor.getScalar(), adaptor.getDest(), offsets, sizes,
        strides);

    rewriter.replaceOp(op, {newOp});
    return mlir::success();
  }
};

/// Rewriter for the `tensor::from_elements` operation.
struct TensorFromElementsOpPattern
    : public CrtOpPattern<mlir::tensor::FromElementsOp> {

  TensorFromElementsOpPattern(mlir::MLIRContext *context,
                              mlir::concretelang::CrtLoweringParameters params,
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
             .isa<FHE::FheIntegerInterface>() &&
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
    auto oldOutputType = outputTensor.getType();
    auto newOutputType = this->getTypeConverter()->convertType(oldOutputType);
    auto newOutputShape =
        newOutputType.cast<mlir::RankedTensorType>().getShape();
    mlir::SmallVector<mlir::OpFoldResult> sizes(newOutputShape.size(),
                                                rewriter.getI64IntegerAttr(1));
    sizes[sizes.size() - 1] =
        rewriter.getI64IntegerAttr(loweringParameters.nMods);
    mlir::SmallVector<mlir::OpFoldResult> strides(
        newOutputShape.size(), rewriter.getI64IntegerAttr(1));

    auto offsetGenerator = [&](size_t index) {
      mlir::SmallVector<mlir::OpFoldResult> offsets(
          newOutputShape.size(), rewriter.getI64IntegerAttr(0));
      size_t remainder = index * 5;
      for (int rankIndex = newOutputShape.size() - 1; rankIndex >= 0;
           --rankIndex) {
        offsets[rankIndex] =
            rewriter.getI64IntegerAttr(remainder % newOutputShape[rankIndex]);
        remainder = remainder / newOutputShape[rankIndex];
      }
      return offsets;
    };

    for (size_t insertionIndex = 0;
         insertionIndex < adaptor.getElements().size(); ++insertionIndex) {
      mlir::tensor::InsertSliceOp insertOp =
          rewriter.create<mlir::tensor::InsertSliceOp>(
              op.getLoc(), adaptor.getElements()[insertionIndex], outputTensor,
              offsetGenerator(insertionIndex), sizes, strides);
      outputTensor = insertOp.getResult();
    }
    rewriter.replaceOp(op, {outputTensor});
    return mlir::success();
  }
};

// Generic template for tensor operations that have reassociation map
// attributes.
template <typename Op, bool inRank>
struct TensorReassociationOpPattern : public CrtOpPattern<Op> {
  TensorReassociationOpPattern(mlir::MLIRContext *context,
                               mlir::concretelang::CrtLoweringParameters params,
                               mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<Op>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::TypeConverter *converter = this->getTypeConverter();

    auto reassocVal = (inRank ? adaptor.getSrc() : op.getResult());
    auto reassocTy = reassocVal.getType();
    auto newReassocType = converter->convertType(reassocTy);

    mlir::SmallVector<mlir::ReassociationIndices> oldReassocs =
        op.getReassociationIndices();
    mlir::SmallVector<mlir::ReassociationIndices> newReassocs{oldReassocs};
    mlir::ReassociationIndices newReassocEnd;
    newReassocEnd.push_back(
        newReassocType.template cast<mlir::RankedTensorType>().getRank() - 1);
    newReassocs.push_back(newReassocEnd);

    auto newOp = rewriter.create<Op>(
        op.getLoc(), converter->convertType(op.getResult().getType()),
        adaptor.getSrc(), newReassocs);
    rewriter.replaceOp(op, {newOp});

    return mlir::success();
  };
};

struct ExtractSliceOpPattern
    : public CrtOpPattern<mlir::tensor::ExtractSliceOp> {
  ExtractSliceOpPattern(mlir::MLIRContext *context,
                        mlir::concretelang::CrtLoweringParameters params,
                        mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<mlir::tensor::ExtractSliceOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                  mlir::tensor::ExtractSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::TypeConverter *converter = this->getTypeConverter();

    mlir::SmallVector<int64_t> newStaticOffsets{op.static_offsets()};
    mlir::SmallVector<int64_t> newStaticSizes{op.static_sizes()};
    mlir::SmallVector<int64_t> newStaticStrides{op.static_strides()};
    newStaticOffsets.push_back(0);
    newStaticSizes.push_back(this->loweringParameters.nMods);
    newStaticStrides.push_back(1);

    mlir::RankedTensorType newType =
        converter->convertType(op.getResult().getType())
            .template cast<mlir::RankedTensorType>();
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        op, newType, adaptor.getSource(), adaptor.getOffsets(),
        adaptor.getSizes(), adaptor.getStrides(),
        rewriter.getDenseI64ArrayAttr(newStaticOffsets),
        rewriter.getDenseI64ArrayAttr(newStaticSizes),
        rewriter.getDenseI64ArrayAttr(newStaticStrides));

    return mlir::success();
  };
};

template <typename OpTy>
struct InsertSliceOpPattern : public CrtOpPattern<OpTy> {
  InsertSliceOpPattern(mlir::MLIRContext *context,
                       mlir::concretelang::CrtLoweringParameters params,
                       mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<OpTy>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // add 0 to offsets
    mlir::SmallVector<mlir::OpFoldResult> offsets = getMixedValues(
        adaptor.getStaticOffsets(), adaptor.getOffsets(), rewriter);
    offsets.push_back(rewriter.getI64IntegerAttr(0));

    // add lweDimension+1 to sizes
    mlir::SmallVector<mlir::OpFoldResult> sizes =
        getMixedValues(adaptor.getStaticSizes(), adaptor.getSizes(), rewriter);
    sizes.push_back(rewriter.getI64IntegerAttr(this->loweringParameters.nMods));

    // add 1 to the strides
    mlir::SmallVector<mlir::OpFoldResult> strides = getMixedValues(
        adaptor.getStaticStrides(), adaptor.getStrides(), rewriter);
    strides.push_back(rewriter.getI64IntegerAttr(1));

    // replace insert slice-like operation with the new one
    rewriter.replaceOpWithNewOp<OpTy>(
        op, adaptor.getSource(), adaptor.getDest(), offsets, sizes, strides);

    return mlir::success();
  };
};

/// Zero op result can be a tensor after CRT encoding, and thus need to be
/// rewritten as a ZeroTensor op
struct ZeroOpPattern : public CrtOpPattern<FHE::ZeroEintOp> {
  ZeroOpPattern(mlir::MLIRContext *context,
                mlir::concretelang::CrtLoweringParameters params,
                mlir::PatternBenefit benefit = 1)
      : CrtOpPattern<FHE::ZeroEintOp>(context, params, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::ZeroEintOp op, FHE::ZeroEintOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::TypeConverter *converter = this->getTypeConverter();
    auto glweOrTensorType = converter->convertType(op.getResult().getType());
    if (mlir::dyn_cast<mlir::TensorType>(glweOrTensorType)) {
      rewriter.replaceOpWithNewOp<TFHE::ZeroTensorGLWEOp>(op, glweOrTensorType);
    } else {
      rewriter.replaceOpWithNewOp<TFHE::ZeroGLWEOp>(op, glweOrTensorType);
    }
    return mlir::success();
  };
};

} // namespace lowering

struct FHEToTFHECrtPass : public FHEToTFHECrtBase<FHEToTFHECrtPass> {

  FHEToTFHECrtPass(mlir::concretelang::CrtLoweringParameters params)
      : loweringParameters(params) {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    typing::TypeConverter converter(loweringParameters);

    //------------------------------------------- Marking legal/illegal dialects
    target.addIllegalDialect<FHE::FHEDialect>();
    target.addLegalDialect<TFHE::TFHEDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
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
    target.addLegalOp<mlir::scf::InParallelOp>();

    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::bufferization::AllocTensorOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::ExtractSliceOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::InsertSliceOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::FromElementsOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::ExpandShapeOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::tensor::CollapseShapeOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<Tracing::TraceCiphertextOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::tensor::ParallelInsertSliceOp>(target, converter);

    //---------------------------------------------------------- Adding patterns
    mlir::RewritePatternSet patterns(&getContext());

    // Patterns for `bufferization` dialect operations.
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::bufferization::AllocTensorOp, true>>(patterns.getContext(),
                                                   converter);

    // Patterns for the `FHE` dialect operations
    patterns.add<lowering::ZeroOpPattern>(&getContext(), loweringParameters);
    patterns.add<
        //    |_ `FHE::zero_tensor`
        mlir::concretelang::GenericOneToOneOpConversionPattern<
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
                 //    |_ `FHE::to_unsigned`
                 lowering::ToUnsignedOpPattern,
                 //    |_ `FHE::to_signed`
                 lowering::ToSignedOpPattern,
                 //    |_ `FHE::apply_lookup_table`
                 lowering::ApplyLookupTableEintOpPattern>(&getContext(),
                                                          loweringParameters);

    // Patterns for the relics of the `FHELinalg` dialect operations.
    //    |_ `linalg::generic` turned to nested `scf::for`
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::scf::ForOp>>(patterns.getContext(), converter);
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::scf::YieldOp>>(patterns.getContext(), converter);
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::scf::ForOp>>(&getContext(), converter);
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::scf::ForallOp>>(&getContext(), converter);

    patterns.add<lowering::TensorExtractOpPattern>(&getContext(),
                                                   loweringParameters);
    patterns.add<lowering::TensorInsertOpPattern>(&getContext(),
                                                  loweringParameters);
    patterns.add<lowering::TensorReassociationOpPattern<
        mlir::tensor::CollapseShapeOp, true>>(patterns.getContext(),
                                              loweringParameters);
    patterns.add<lowering::TensorReassociationOpPattern<
        mlir::tensor::ExpandShapeOp, false>>(patterns.getContext(),
                                             loweringParameters);
    patterns.add<lowering::ExtractSliceOpPattern>(patterns.getContext(),
                                                  loweringParameters);
    patterns.add<
        lowering::InsertSliceOpPattern<mlir::tensor::InsertSliceOp>,
        lowering::InsertSliceOpPattern<mlir::tensor::ParallelInsertSliceOp>>(
        patterns.getContext(), loweringParameters);
    patterns.add<lowering::TraceCiphertextOpPattern>(patterns.getContext(),
                                                     loweringParameters);
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::tensor::GenerateOp, true>>(&getContext(), converter);

    // Patterns for `func` dialect operations.
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::func::ReturnOp>>(patterns.getContext(), converter);
    patterns.add<FunctionConstantOpConversion<typing::TypeConverter>>(
        &getContext(), converter);

    // Pattern for the `tensor::from_element` op.
    patterns.add<lowering::TensorFromElementsOpPattern>(patterns.getContext(),
                                                        loweringParameters);

    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::scf::YieldOp>>(&getContext(), converter);

    mlir::concretelang::populateWithRTTypeConverterPatterns(patterns, target,
                                                            converter);
    patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
        mlir::scf::ForallOp>>(&getContext(), converter);

    //--------------------------------------------------------- Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }

private:
  mlir::concretelang::CrtLoweringParameters loweringParameters;
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
