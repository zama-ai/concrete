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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/FHEToTFHEScalar/Pass.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
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
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"

namespace FHE = mlir::concretelang::FHE;
namespace TFHE = mlir::concretelang::TFHE;
namespace Tracing = mlir::concretelang::Tracing;
namespace concretelang = mlir::concretelang;

namespace fhe_to_tfhe_scalar_conversion {

namespace typing {

/// Converts an encrypted integer into `TFHE::GlweCiphetext`.
TFHE::GLWECipherTextType convertEncrypted(mlir::MLIRContext *context,
                                          FHE::FheIntegerInterface enc) {
  return TFHE::GLWECipherTextType::get(context, -1, -1, -1, enc.getWidth());
}

/// Converts `Tensor<FHE::AnyEncryptedInteger>` into a
/// `Tensor<TFHE::GlweCiphertext>` if the element type is appropriate.
/// Otherwise return the input type.
mlir::Type
maybeConvertEncryptedTensor(mlir::MLIRContext *context,
                            mlir::RankedTensorType maybeEncryptedTensor) {
  if (!maybeEncryptedTensor.getElementType().isa<FHE::FheIntegerInterface>()) {
    return (mlir::Type)(maybeEncryptedTensor);
  }
  auto enc =
      maybeEncryptedTensor.getElementType().cast<FHE::FheIntegerInterface>();
  auto currentShape = maybeEncryptedTensor.getShape();
  return mlir::RankedTensorType::get(
      currentShape,
      TFHE::GLWECipherTextType::get(context, -1, -1, -1, enc.getWidth()));
}

/// Converts any encrypted type to `TFHE::GlweCiphetext` if the
/// input type is appropriate. Otherwise return the input type.
mlir::Type maybeConvertEncrypted(mlir::MLIRContext *context, mlir::Type t) {
  if (auto eint = t.dyn_cast<FHE::FheIntegerInterface>())
    return convertEncrypted(context, eint);
  return t;
}

/// The type converter used to convert `FHE` to `TFHE` types using the scalar
/// strategy.
class TypeConverter : public mlir::TypeConverter {

public:
  TypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([](FHE::FheIntegerInterface type) {
      return convertEncrypted(type.getContext(), type);
    });
    addConversion([](FHE::EncryptedBooleanType type) {
      return TFHE::GLWECipherTextType::get(
          type.getContext(), -1, -1, -1,
          mlir::concretelang::FHE::EncryptedBooleanType::getWidth());
    });
    addConversion([](mlir::RankedTensorType type) {
      return maybeConvertEncryptedTensor(type.getContext(), type);
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
struct ScalarOpPattern : public mlir::OpConversionPattern<T> {

  ScalarOpPattern(mlir::TypeConverter &converter, mlir::MLIRContext *context,
                  mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<T>(converter, context, benefit) {}

  /// Writes the encoding of a plaintext of arbitrary precision using shift.
  mlir::Value
  writePlaintextShiftEncoding(mlir::Location location, mlir::Value rawPlaintext,
                              int64_t encryptedWidth,
                              mlir::ConversionPatternRewriter &rewriter) const {
    int64_t intShift = 64 - 1 - encryptedWidth;
    mlir::Value castedInt = rewriter.create<mlir::arith::ExtSIOp>(
        location, rewriter.getIntegerType(64), rawPlaintext);
    mlir::Value constantShiftOp = rewriter.create<mlir::arith::ConstantOp>(
        location, rewriter.getI64IntegerAttr(intShift));
    mlir::Value encodedInt = rewriter.create<mlir::arith::ShLIOp>(
        location, rewriter.getI64Type(), castedInt, constantShiftOp);
    return encodedInt;
  }
};

/// Rewriter for the `FHE::add_eint_int` operation.
struct AddEintIntOpPattern : public ScalarOpPattern<FHE::AddEintIntOp> {
  AddEintIntOpPattern(mlir::TypeConverter &converter,
                      mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::AddEintIntOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::AddEintIntOp op, FHE::AddEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Write the plaintext encoding
    mlir::Value encodedInt = writePlaintextShiftEncoding(
        op.getLoc(), adaptor.b(),
        op.getType().cast<FHE::FheIntegerInterface>().getWidth(), rewriter);

    // Write the new op
    rewriter.replaceOpWithNewOp<TFHE::AddGLWEIntOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.a(),
        encodedInt);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::sub_eint_int` operation.
struct SubEintIntOpPattern : public ScalarOpPattern<FHE::SubEintIntOp> {
  SubEintIntOpPattern(mlir::TypeConverter &converter,
                      mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::SubEintIntOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::SubEintIntOp op, FHE::SubEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = op.a();
    mlir::Value intOperand = op.b();

    // Write the integer negation
    mlir::Type intType = intOperand.getType();
    mlir::Attribute minusOneAttr = mlir::IntegerAttr::get(intType, -1);
    mlir::Value minusOne =
        rewriter.create<mlir::arith::ConstantOp>(location, minusOneAttr)
            .getResult();
    mlir::Value negative =
        rewriter.create<mlir::arith::MulIOp>(location, intOperand, minusOne)
            .getResult();

    // Write the plaintext encoding
    mlir::Value encodedInt = writePlaintextShiftEncoding(
        op.getLoc(), negative,
        eintOperand.getType().cast<FHE::FheIntegerInterface>().getWidth(),
        rewriter);

    // Write the new op
    rewriter.replaceOpWithNewOp<TFHE::AddGLWEIntOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.a(),
        encodedInt);

    return mlir::success();
  };
};

/// Rewriter for the `FHE::sub_int_eint` operation.
struct SubIntEintOpPattern : public ScalarOpPattern<FHE::SubIntEintOp> {
  SubIntEintOpPattern(mlir::TypeConverter &converter,
                      mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::SubIntEintOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::SubIntEintOp op, FHE::SubIntEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Write the plaintext encoding
    mlir::Value encodedInt = writePlaintextShiftEncoding(
        op.getLoc(), adaptor.a(),
        op.b().getType().cast<FHE::FheIntegerInterface>().getWidth(), rewriter);

    // Write the new op
    rewriter.replaceOpWithNewOp<TFHE::SubGLWEIntOp>(
        op, getTypeConverter()->convertType(op.getType()), encodedInt,
        adaptor.b());

    return mlir::success();
  };
};

/// Rewriter for the `FHE::sub_eint` operation.
struct SubEintOpPattern : public ScalarOpPattern<FHE::SubEintOp> {
  SubEintOpPattern(mlir::TypeConverter &converter, mlir::MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::SubEintOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::SubEintOp op, FHE::SubEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location location = op.getLoc();
    mlir::Value lhsOperand = adaptor.a();
    mlir::Value rhsOperand = adaptor.b();

    // Write rhs negation
    auto negative = rewriter.create<TFHE::NegGLWEOp>(
        location, rhsOperand.getType(), rhsOperand);

    // Write new op.
    rewriter.replaceOpWithNewOp<TFHE::AddGLWEOp>(
        op, getTypeConverter()->convertType(op.getType()), lhsOperand,
        negative.getResult());

    return mlir::success();
  };
};

/// Rewriter for the `FHE::mul_eint_int` operation.
struct MulEintIntOpPattern : public ScalarOpPattern<FHE::MulEintIntOp> {
  MulEintIntOpPattern(mlir::TypeConverter &converter,
                      mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::MulEintIntOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::MulEintIntOp op, FHE::MulEintIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location location = op.getLoc();
    mlir::Value eintOperand = adaptor.a();
    mlir::Value intOperand = adaptor.b();

    // Write the cleartext "encoding"
    mlir::Value castedCleartext = rewriter.create<mlir::arith::ExtSIOp>(
        location, rewriter.getIntegerType(64), intOperand);

    // Write the new op.
    rewriter.replaceOpWithNewOp<TFHE::MulGLWEIntOp>(
        op, getTypeConverter()->convertType(op.getType()), eintOperand,
        castedCleartext);

    return mlir::success();
  }
};

/// Rewriter for the `FHE::to_signed` operation.
struct ToSignedOpPattern : public ScalarOpPattern<FHE::ToSignedOp> {
  ToSignedOpPattern(mlir::TypeConverter &converter, mlir::MLIRContext *context,
                    mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::ToSignedOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ToSignedOp op, FHE::ToSignedOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    typing::TypeConverter converter;
    rewriter.replaceOp(op, {adaptor.input()});

    return mlir::success();
  }
};

/// Rewriter for the `FHE::to_unsigned` operation.
struct ToUnsignedOpPattern : public ScalarOpPattern<FHE::ToUnsignedOp> {
  ToUnsignedOpPattern(mlir::TypeConverter &converter,
                      mlir::MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::ToUnsignedOp>(converter, context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ToUnsignedOp op, FHE::ToUnsignedOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    typing::TypeConverter converter;
    rewriter.replaceOp(op, {adaptor.input()});

    return mlir::success();
  }
};

/// Rewriter for the `FHE::apply_lookup_table` operation.
struct ApplyLookupTableEintOpPattern
    : public ScalarOpPattern<FHE::ApplyLookupTableEintOp> {
  ApplyLookupTableEintOpPattern(
      mlir::TypeConverter &converter, mlir::MLIRContext *context,
      concretelang::ScalarLoweringParameters loweringParams,
      mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::ApplyLookupTableEintOp>(converter, context,
                                                     benefit),
        loweringParameters(loweringParams) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ApplyLookupTableEintOp op,
                  FHE::ApplyLookupTableEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto inputType = op.a().getType().cast<FHE::FheIntegerInterface>();
    size_t outputBits =
        op.getResult().getType().cast<FHE::FheIntegerInterface>().getWidth();
    mlir::Value newLut =
        rewriter
            .create<TFHE::EncodeExpandLutForBootstrapOp>(
                op.getLoc(),
                mlir::RankedTensorType::get(
                    mlir::ArrayRef<int64_t>(loweringParameters.polynomialSize),
                    rewriter.getI64Type()),
                op.lut(),
                rewriter.getI32IntegerAttr(loweringParameters.polynomialSize),
                rewriter.getI32IntegerAttr(outputBits),
                rewriter.getBoolAttr(inputType.isSigned()))
            .getResult();

    typing::TypeConverter converter;
    mlir::Value input = adaptor.a();

    if (inputType.isSigned()) {
      // If the input is a signed integer, it comes to the bootstrap with a
      // signed-leveled encoding (compatible with 2s complement semantics).
      // Unfortunately pbs is not compatible with this encoding, since the
      // (virtual) msb must be 0 to avoid a lookup in the phantom negative lut.
      uint64_t constantRaw = (uint64_t)1 << (inputType.getWidth() - 1);
      // Note that the constant must be encoded with one more bit to ensure the
      // signed extension used in the plaintext encoding works as expected.
      mlir::Value constant = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(inputType.getWidth() + 1), constantRaw));
      mlir::Value encodedConstant = writePlaintextShiftEncoding(
          op.getLoc(), constant, inputType.getWidth(), rewriter);
      auto inputOp = rewriter.create<TFHE::AddGLWEIntOp>(
          op.getLoc(), converter.convertType(input.getType()), input,
          encodedConstant);
      input = inputOp;
    }

    // Insert keyswitch
    auto ksOp = rewriter.create<TFHE::KeySwitchGLWEOp>(
        op.getLoc(), getTypeConverter()->convertType(adaptor.a().getType()),
        input, -1, -1);

    // Insert bootstrap
    rewriter.replaceOpWithNewOp<TFHE::BootstrapGLWEOp>(
        op, getTypeConverter()->convertType(op.getType()), ksOp, newLut, -1, -1,
        -1, -1);

    return mlir::success();
  };

private:
  concretelang::ScalarLoweringParameters loweringParameters;
};

struct RoundEintOpPattern : public ScalarOpPattern<FHE::RoundEintOp> {
  RoundEintOpPattern(mlir::TypeConverter &converter, mlir::MLIRContext *context,
                     concretelang::ScalarLoweringParameters loweringParams,
                     mlir::PatternBenefit benefit = 1)
      : ScalarOpPattern<FHE::RoundEintOp>(converter, context, benefit),
        loweringParameters(loweringParams) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHE::RoundEintOp op, FHE::RoundEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // The round operator allows to move from a given precision to a smaller one
    // by rounding the most significant bits of the message. For example a 5
    // bits message:
    //      101_11 (23)
    // would be rounded to a 3 bit message:
    //      110    (6)
    //
    // The following procedure can be homomorphically applied to implement this
    // semantic:
    //      1) Propagate the carry of the round around 2^(n_before-n_after)
    //         performed with a homomorphic adddition.
    //      2) For each bits to be discarded we truncate it:
    //             -> Extract a ciphertext of only the bit to be discarded by
    //                performing a left shift and a pbs.
    //             -> Subtract this one from the input by performing a
    //                homomorphic subtraction.

    mlir::Value input = adaptor.input();
    auto inputType = op.input().getType().cast<FHE::FheIntegerInterface>();
    mlir::Value output = op.getResult();
    uint64_t inputBitwidth = inputType.getWidth();
    uint64_t outputBitwidth =
        output.getType().cast<FHE::FheIntegerInterface>().getWidth();
    uint64_t bitwidthDelta = inputBitwidth - outputBitwidth;

    typing::TypeConverter converter;
    auto inputTy =
        converter.convertType(inputType).cast<TFHE::GLWECipherTextType>();

    //-------------------------------------------------------- CARRY PROPAGATION
    // The first step we take is to propagate the carry of the round in the
    // msbs. This we perform with an addition of cleartext correctly encoded.
    // Say we have a 5 bits message that we want to round for 3 bits, we
    // perform the following addition:
    //
    //     input            = |0101|11| .... |
    //     carryCst         = |0000|10| .... |
    //     input + carryCst = |0110|01| .... |

    uint64_t rawCarryCst = ((uint64_t)1) << (bitwidthDelta - 1);
    mlir::Value carryCst = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(bitwidthDelta + 1),
                                rawCarryCst));
    mlir::Value encodedCarryCst = writePlaintextShiftEncoding(
        op.getLoc(), carryCst, inputBitwidth, rewriter);
    mlir::Value carryPropagatedVal = rewriter.create<TFHE::AddGLWEIntOp>(
        op.getLoc(), inputTy, input, encodedCarryCst);

    //--------------------------------------------------------------- TRUNCATION
    // The second step is to truncate every lsbs to be removed, from the least
    // significant one to the most significant one. For example:
    //
    //     previousOutput = |0110|01| .... | (t_0)
    //     previousOutput = |0110|00| .... | (t_1)
    //                             ^
    //     previousOutput = |0110|00| .... | (t_2)
    //                            ^
    //
    // For this, we have to generate a ciphertext that contains only the bit to
    // be truncated:
    //
    //     bitToRemove = |0000|01| .... | (t_1)
    //                          ^
    //     bitToRemove = |0000|00| .... | (t_1)
    //                         ^

    mlir::Value previousOutput = carryPropagatedVal;
    TFHE::GLWECipherTextType truncationInputTy = inputTy;
    for (uint64_t i = 0; i < bitwidthDelta; ++i) {
      //---------------------------------------------------------- BIT ISOLATION
      // To extract the bit to truncate, we use a PBS that look up on the
      // padding bit. We first begin by isolating the bit in question on the
      // padding bit. This is performed with a homomorphic multiplication (left
      // shift basically) of the proper amount. For example:
      //
      //     previousOutput            = |0110|01| .... |
      //                                        ^
      //     shiftCst                  = |        100000|
      //     previousOutput * shiftCst = |1| ....       |

      uint64_t rawShiftCst = ((uint64_t)1) << (inputBitwidth - i);
      mlir::Value shiftCst = rewriter.create<mlir::arith::ConstantOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(rawShiftCst));
      mlir::Value shiftedInput = rewriter.create<TFHE::MulGLWEIntOp>(
          op.getLoc(), truncationInputTy, previousOutput, shiftCst);

      //-------------------------------------------------------- LUT PREPARATION
      // To perform the right shift (kind of), we use a PBS that acts on the
      // padding bit. We expect is the following function to be applied (for the
      // first round of our example):
      //
      //     f(|0| .... |) = |0000|00| .... |
      //     f(|1| .... |) = |0000|01| .... |
      //
      // That being said, a PBS on the padding bit can only encode a symmetric
      // function (that is f(1) = -f(0)), by encoding f(0) in the whole table.
      // To implement our semantic, we then rely on a trick. We encode the
      // following function in the bootstrap:
      //
      //     f(|0| .... |) = |1111|11|1 .... |
      //     f(|1| .... |) = |0000|00|1 .... |
      //
      // And add a correction constant:
      //
      //     corrCst                 = |0000|00|1 .... |
      //     f(|0| .... |) + corrCst = |0000|00| .... |
      //     f(|1| .... |) + corrCst = |0000|01| .... |
      //
      // Hence the following constant lut.

      llvm::SmallVector<int64_t> rawLut(loweringParameters.polynomialSize,
                                        ((uint64_t)0 - 1)
                                            << (64 - (inputBitwidth + 2 - i)));
      mlir::Value lut = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), mlir::DenseIntElementsAttr::get(
                           mlir::RankedTensorType::get(
                               rawLut.size(), rewriter.getIntegerType(64)),
                           rawLut));

      //-------------------------------------------------- CIPHERTEXT ALIGNEMENT
      // In practice, TFHE ciphertexts are normally distributed around a value.
      // That means that if the lookup is performed _as is_, we have almost .5
      // probability to return the wrong value. Imagine a ciphertext centered
      // around (|0| .... |):
      //
      //  |   0000001...  |   1111111...  |              Virtual lookup table
      //                   _
      //                  / \
      //  _______________/   \_________________________  Ciphertext distribution
      //
      //              |0| ... |                          Ciphertexts mean
      //
      // If the error of the ciphertext is negative, this means that the lookup
      // will wrap, and fall on the wrong mega-case...
      //
      // This is usually taken care of on the lookup table side, but we can also
      // slightly shift the ciphertext to center its distribution with the
      // center of the mega-case. That is, end up with a situation like this:

      //
      //  |   1111111...  |   0000001...  |              Virtual lookup table
      //          _
      //         / \
      //  ______/   \_________________________           Ciphertext distribution
      //
      //      |0| ... |                                  Ciphertexts mean
      //
      // This is performed by adding |0|1 .... | to the ciphertext.

      uint64_t rawRotationCst = (((uint64_t)1) << 62);
      mlir::Value rotationCst = rewriter.create<mlir::arith::ConstantOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(rawRotationCst));
      mlir::Value shiftedRotatedInput = rewriter.create<TFHE::AddGLWEIntOp>(
          op.getLoc(), truncationInputTy, shiftedInput, rotationCst);

      //-------------------------------------------------------------------- PBS
      // The lookup is performed ...

      mlir::Value keyswitched = rewriter.create<TFHE::KeySwitchGLWEOp>(
          op.getLoc(), truncationInputTy, shiftedRotatedInput, -1, -1);
      mlir::Value bootstrapped = rewriter.create<TFHE::BootstrapGLWEOp>(
          op.getLoc(), truncationInputTy, keyswitched, lut, -1, -1, -1, -1);

      //------------------------------------------------------------- CORRECTION
      // The correction is performed to achieve our right shift semantic.

      uint64_t rawCorrCst = ((uint64_t)1) << (64 - (inputBitwidth + 2 - i));
      mlir::Value corrCst = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(rawCorrCst));
      mlir::Value extractedBit = rewriter.create<TFHE::AddGLWEIntOp>(
          op.getLoc(), truncationInputTy, bootstrapped, corrCst);

      //------------------------------------------------------------- TRUNCATION
      // Finally, the extracted bit is subtracted from the input.

      mlir::Value minusIsolatedBit = rewriter.create<TFHE::NegGLWEOp>(
          op.getLoc(), truncationInputTy, extractedBit);
      truncationInputTy = TFHE::GLWECipherTextType::get(
          rewriter.getContext(), -1, -1, -1, truncationInputTy.getP() - 1);
      mlir::Value truncationOutput = rewriter.create<TFHE::AddGLWEOp>(
          op.getLoc(), truncationInputTy, previousOutput, minusIsolatedBit);
      previousOutput = truncationOutput;
    }

    rewriter.replaceOp(op, {previousOutput});

    return mlir::success();
  };

private:
  concretelang::ScalarLoweringParameters loweringParameters;
};

/// Rewriter for the `FHE::to_bool` operation.
struct ToBoolOpPattern : public mlir::OpRewritePattern<FHE::ToBoolOp> {
  ToBoolOpPattern(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<FHE::ToBoolOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ToBoolOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto width = op.input()
                     .getType()
                     .dyn_cast<mlir::concretelang::FHE::EncryptedIntegerType>()
                     .getWidth();
    if (width == mlir::concretelang::FHE::EncryptedBooleanType::getWidth()) {
      rewriter.replaceOp(op, op.input());
      return mlir::success();
    }
    // TODO
    op->emitError("only support conversion with width 2 for the moment");
    return mlir::failure();
  }
};

/// Rewriter for the `FHE::from_bool` operation.
struct FromBoolOpPattern : public mlir::OpRewritePattern<FHE::FromBoolOp> {
  FromBoolOpPattern(mlir::MLIRContext *context,
                    mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<FHE::FromBoolOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::FromBoolOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto width = op.getResult()
                     .getType()
                     .dyn_cast<mlir::concretelang::FHE::EncryptedIntegerType>()
                     .getWidth();
    if (width == mlir::concretelang::FHE::EncryptedBooleanType::getWidth()) {
      rewriter.replaceOp(op, op.input());
      return mlir::success();
    }
    // TODO
    op->emitError("only support conversion with width 2 for the moment");
    return mlir::failure();
  }
};

} // namespace lowering

struct FHEToTFHEScalarPass : public FHEToTFHEScalarBase<FHEToTFHEScalarPass> {

  FHEToTFHEScalarPass(concretelang::ScalarLoweringParameters loweringParams)
      : loweringParameters(loweringParams){};

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    typing::TypeConverter converter;

    //------------------------------------------- Marking legal/illegal dialects
    target.addIllegalDialect<FHE::FHEDialect>();
    target.addLegalDialect<TFHE::TFHEDialect>();
    target.addLegalDialect<mlir::arith::ArithmeticDialect>();
    target.addDynamicallyLegalOp<mlir::linalg::GenericOp,
                                 mlir::tensor::GenerateOp>(
        [&](mlir::Operation *op) {
          return (
              converter.isLegal(op->getOperandTypes()) &&
              converter.isLegal(op->getResultTypes()) &&
              converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
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

    // Patterns for the `FHE` dialect operations
    patterns.add<
        //    |_ `FHE::zero_eint`
        concretelang::GenericOneToOneOpConversionPattern<FHE::ZeroEintOp,
                                                         TFHE::ZeroGLWEOp>,
        //    |_ `FHE::zero_tensor`
        concretelang::GenericOneToOneOpConversionPattern<
            FHE::ZeroTensorOp, TFHE::ZeroTensorGLWEOp>,
        //    |_ `FHE::neg_eint`
        concretelang::GenericOneToOneOpConversionPattern<FHE::NegEintOp,
                                                         TFHE::NegGLWEOp>,
        //    |_ `FHE::not`
        concretelang::GenericOneToOneOpConversionPattern<FHE::BoolNotOp,
                                                         TFHE::NegGLWEOp>,
        //    |_ `FHE::add_eint`
        concretelang::GenericOneToOneOpConversionPattern<FHE::AddEintOp,
                                                         TFHE::AddGLWEOp>>(
        &getContext(), converter);
    //    |_ `FHE::add_eint_int`
    patterns.add<lowering::AddEintIntOpPattern,
                 //    |_ `FHE::sub_int_eint`
                 lowering::SubIntEintOpPattern,
                 //    |_ `FHE::sub_eint_int`
                 lowering::SubEintIntOpPattern,
                 //    |_ `FHE::sub_eint`
                 lowering::SubEintOpPattern,
                 //    |_ `FHE::mul_eint_int`
                 lowering::MulEintIntOpPattern,
                 //    |_ `FHE::to_signed`
                 lowering::ToSignedOpPattern,
                 //    |_ `FHE::to_unsigned`
                 lowering::ToUnsignedOpPattern>(converter, &getContext());
    //    |_ `FHE::apply_lookup_table`
    patterns.add<lowering::ApplyLookupTableEintOpPattern,
                 //    |_ `FHE::round`
                 lowering::RoundEintOpPattern>(converter, &getContext(),
                                               loweringParameters);

    // Patterns for boolean conversion ops
    patterns.add<lowering::FromBoolOpPattern, lowering::ToBoolOpPattern>(
        &getContext());

    // Patterns for the relics of the `FHELinalg` dialect operations.
    //    |_ `linalg::generic` turned to nested `scf::for`
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::linalg::YieldOp>>(patterns.getContext(), converter);
    patterns.add<
        concretelang::TypeConvertingReinstantiationPattern<
            mlir::tensor::GenerateOp, true>,
        concretelang::TypeConvertingReinstantiationPattern<mlir::scf::ForOp>>(
        &getContext(), converter);
    concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                          converter);

    // Patterns for `func` dialect operations.
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::func::ReturnOp>>(patterns.getContext(), converter);

    concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(target,
                                                                  converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::scf::YieldOp>(target,
                                                                converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::scf::ForOp>(target,
                                                              converter);
    patterns.add<FunctionConstantOpConversion<typing::TypeConverter>>(
        &getContext(), converter);

    // Patterns for `tracing` dialect.
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        Tracing::TraceCiphertextOp, true>>(&getContext(), converter);
    concretelang::addDynamicallyLegalTypeOp<Tracing::TraceCiphertextOp>(
        target, converter);

    // Patterns for `bufferization` dialect operations.
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::bufferization::AllocTensorOp, true>>(patterns.getContext(),
                                                   converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::bufferization::AllocTensorOp>(
        target, converter);

    // Patterns for the `RT` dialect operations.
    patterns.add<
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
  concretelang::ScalarLoweringParameters loweringParameters;
};

} // namespace fhe_to_tfhe_scalar_conversion

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertFHEToTFHEScalarPass(ScalarLoweringParameters loweringParameters) {
  return std::make_unique<fhe_to_tfhe_scalar_conversion::FHEToTFHEScalarPass>(
      loweringParameters);
}
} // namespace concretelang
} // namespace mlir
