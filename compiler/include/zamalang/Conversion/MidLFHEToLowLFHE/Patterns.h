#ifndef ZAMALANG_CONVERSION_MIDLFHETOLOWLFHE_PATTERNS_H_
#define ZAMALANG_CONVERSION_MIDLFHETOLOWLFHE_PATTERNS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"

namespace mlir {
namespace zamalang {

using LowLFHE::CleartextType;
using LowLFHE::LweCiphertextType;
using LowLFHE::PlaintextType;
using MidLFHE::GLWECipherTextType;

LweCiphertextType convertTypeToLWE(mlir::MLIRContext *context,
                                   mlir::Type type) {
  auto glwe = type.dyn_cast_or_null<GLWECipherTextType>();
  if (glwe != nullptr) {
    return LweCiphertextType::get(
        context, glwe.getDimension() * glwe.getPolynomialSize(), glwe.getP());
  }
  auto lwe = type.dyn_cast_or_null<LweCiphertextType>();
  if (lwe != nullptr) {
    return lwe;
  }
  assert(false && "expect glwe or lwe");
}

template <typename PType>
PlaintextType convertPlaintextTypeFromPType(mlir::MLIRContext *context,
                                            PType &type) {
  return PlaintextType::get(context, type.getP() + 1);
}

// convertPlaintextTypeFromType create a plaintext type according the
// precision of the given type argument. The type should be a GLWECipherText
// (if operand is not yet lowered) or a LWECipherTextType (if operand is
// already lowered).
PlaintextType convertPlaintextTypeFromType(mlir::MLIRContext *context,
                                           mlir::Type &type) {
  auto glwe = type.dyn_cast_or_null<GLWECipherTextType>();
  if (glwe != nullptr) {
    return convertPlaintextTypeFromPType<GLWECipherTextType>(context, glwe);
  }
  auto lwe = type.dyn_cast_or_null<LweCiphertextType>();
  if (lwe != nullptr) {
    return convertPlaintextTypeFromPType<LweCiphertextType>(context, lwe);
  }
  assert(false && "expect glwe or lwe");
}

template <typename PType>
CleartextType convertCleartextTypeFromPType(mlir::MLIRContext *context,
                                            PType &type) {
  return CleartextType::get(context, type.getP() + 1);
}

// convertCleartextTypeFromType create a cleartext type according the
// precision of the given type argument. The type should be a GLWECipherText
// (if operand is not yet lowered) or a LWECipherTextType (if operand is
// already lowered).
CleartextType convertCleartextTypeFromType(mlir::MLIRContext *context,
                                           mlir::Type &type) {
  auto glwe = type.dyn_cast_or_null<GLWECipherTextType>();
  if (glwe != nullptr) {
    return convertCleartextTypeFromPType<GLWECipherTextType>(context, glwe);
  }
  auto lwe = type.dyn_cast_or_null<LweCiphertextType>();
  if (lwe != nullptr) {
    return convertCleartextTypeFromPType<LweCiphertextType>(context, lwe);
  }
  assert(false && "expect glwe or lwe");
}

mlir::Value createZeroLWEOpFromMidLFHE(mlir::PatternRewriter rewriter,
                                       mlir::Location loc,
                                       mlir::OpResult result) {
  mlir::SmallVector<mlir::Value> args{};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto glwe = result.getType().cast<GLWECipherTextType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeToLWE(rewriter.getContext(), glwe)};
  LowLFHE::ZeroLWEOp op =
      rewriter.create<LowLFHE::ZeroLWEOp>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createLowLFHEOpFromMidLFHE(mlir::PatternRewriter rewriter,
                                       mlir::Location loc, mlir::Value arg0,
                                       mlir::Value arg1,
                                       mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 2> args{arg0, arg1};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto glwe = result.getType().cast<GLWECipherTextType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeToLWE(rewriter.getContext(), glwe)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

mlir::Value createAddPlainLweCiphertextWithGlwe(
    mlir::PatternRewriter rewriter, mlir::Location loc, mlir::Value arg0,
    mlir::Value arg1, mlir::OpResult result, mlir::Type encryptedType) {
  PlaintextType encoded_type =
      convertPlaintextTypeFromType(rewriter.getContext(), encryptedType);
  // encode int into plaintext
  mlir::Value encoded =
      rewriter
          .create<mlir::zamalang::LowLFHE::EncodeIntOp>(loc, encoded_type, arg1)
          .plaintext();
  // convert result type
  GLWECipherTextType glwe_type = result.getType().cast<GLWECipherTextType>();
  LweCiphertextType lwe_type =
      convertTypeToLWE(rewriter.getContext(), result.getType());
  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter.create<mlir::zamalang::LowLFHE::AddPlaintextLweCiphertextOp>(
          loc, lwe_type, arg0, encoded);
  return op.getODSResults(0).front();
}

mlir::Value createAddPlainLweCiphertext(mlir::PatternRewriter rewriter,
                                        mlir::Location loc, mlir::Value arg0,
                                        mlir::Value arg1,
                                        mlir::OpResult result) {
  auto glwe = arg0.getType().cast<GLWECipherTextType>();
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, arg0, arg1, result,
                                             glwe);
}

mlir::Value createSubIntLweCiphertext(mlir::PatternRewriter rewriter,
                                      mlir::Location loc, mlir::Value arg0,
                                      mlir::Value arg1, mlir::OpResult result) {
  auto arg1_type = arg1.getType();
  auto negated_arg1 =
      rewriter
          .create<mlir::zamalang::LowLFHE::NegateLweCiphertextOp>(
              loc, convertTypeToLWE(rewriter.getContext(), arg1_type), arg1)
          .result();
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, negated_arg1, arg0,
                                             result, arg1_type);
}

mlir::Value createMulClearLweCiphertext(mlir::PatternRewriter rewriter,
                                        mlir::Location loc, mlir::Value arg0,
                                        mlir::Value arg1,
                                        mlir::OpResult result) {
  auto inType = arg0.getType();
  CleartextType encoded_type =
      convertCleartextTypeFromType(rewriter.getContext(), inType);
  // encode int into plaintext
  mlir::Value encoded = rewriter
                            .create<mlir::zamalang::LowLFHE::IntToCleartextOp>(
                                loc, encoded_type, arg1)
                            .cleartext();
  // convert result type
  auto resType = result.getType();
  LweCiphertextType lwe_type = convertTypeToLWE(rewriter.getContext(), resType);
  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter.create<mlir::zamalang::LowLFHE::MulCleartextLweCiphertextOp>(
          loc, lwe_type, arg0, encoded);
  return op.getODSResults(0).front();
}

} // namespace zamalang
} // namespace mlir

namespace {
#include "zamalang/Conversion/MidLFHEToLowLFHE/Patterns.h.inc"
}

void populateWithGeneratedMidLFHEToLowLFHE(mlir::RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

#endif
