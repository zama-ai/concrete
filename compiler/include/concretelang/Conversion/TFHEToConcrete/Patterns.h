// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_TFHETOCONCRETE_PATTERNS_H_
#define CONCRETELANG_CONVERSION_TFHETOCONCRETE_PATTERNS_H_

#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace concretelang {

using Concrete::CleartextType;
using Concrete::LweCiphertextType;
using Concrete::PlaintextType;
using TFHE::GLWECipherTextType;

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
  return nullptr;
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
  return nullptr;
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
  return nullptr;
}

mlir::Value createZeroLWEOpFromTFHE(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value> args{};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto glwe = result.getType().cast<GLWECipherTextType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeToLWE(rewriter.getContext(), glwe)};
  Concrete::ZeroLWEOp op =
      rewriter.create<Concrete::ZeroLWEOp>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createConcreteOpFromTFHE(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value arg0,
                                     mlir::Value arg1, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 2> args{arg0, arg1};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto glwe = result.getType().cast<GLWECipherTextType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeToLWE(rewriter.getContext(), glwe)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

mlir::Value createAddPlainLweCiphertextWithGlwe(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value arg0,
    mlir::Value arg1, mlir::OpResult result, mlir::Type encryptedType) {
  PlaintextType encoded_type =
      convertPlaintextTypeFromType(rewriter.getContext(), encryptedType);
  // encode int into plaintext
  mlir::Value encoded = rewriter
                            .create<mlir::concretelang::Concrete::EncodeIntOp>(
                                loc, encoded_type, arg1)
                            .plaintext();
  // convert result type
  LweCiphertextType lwe_type =
      convertTypeToLWE(rewriter.getContext(), result.getType());
  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter
          .create<mlir::concretelang::Concrete::AddPlaintextLweCiphertextOp>(
              loc, lwe_type, arg0, encoded);
  return op.getODSResults(0).front();
}

mlir::Value createAddPlainLweCiphertext(mlir::PatternRewriter &rewriter,
                                        mlir::Location loc, mlir::Value arg0,
                                        mlir::Value arg1,
                                        mlir::OpResult result) {
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, arg0, arg1, result,
                                             arg0.getType());
}

mlir::Value createSubIntLweCiphertext(mlir::PatternRewriter &rewriter,
                                      mlir::Location loc, mlir::Value arg0,
                                      mlir::Value arg1, mlir::OpResult result) {
  auto arg1_type = arg1.getType();
  auto negated_arg1 =
      rewriter
          .create<mlir::concretelang::Concrete::NegateLweCiphertextOp>(
              loc, convertTypeToLWE(rewriter.getContext(), arg1_type), arg1)
          .result();
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, negated_arg1, arg0,
                                             result, arg1_type);
}

mlir::Value createNegLweCiphertext(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Value arg0,
                                   mlir::OpResult result) {
  auto arg0_type = arg0.getType();
  auto negated =
      rewriter.create<mlir::concretelang::Concrete::NegateLweCiphertextOp>(
          loc, convertTypeToLWE(rewriter.getContext(), arg0_type), arg0);
  return negated.getODSResults(0).front();
}

mlir::Value createMulClearLweCiphertext(mlir::PatternRewriter &rewriter,
                                        mlir::Location loc, mlir::Value arg0,
                                        mlir::Value arg1,
                                        mlir::OpResult result) {
  auto inType = arg0.getType();
  CleartextType encoded_type =
      convertCleartextTypeFromType(rewriter.getContext(), inType);
  // encode int into plaintext
  mlir::Value encoded =
      rewriter
          .create<mlir::concretelang::Concrete::IntToCleartextOp>(
              loc, encoded_type, arg1)
          .cleartext();
  // convert result type
  auto resType = result.getType();
  LweCiphertextType lwe_type = convertTypeToLWE(rewriter.getContext(), resType);
  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter
          .create<mlir::concretelang::Concrete::MulCleartextLweCiphertextOp>(
              loc, lwe_type, arg0, encoded);
  return op.getODSResults(0).front();
}

} // namespace concretelang
} // namespace mlir

namespace {
#include "concretelang/Conversion/TFHEToConcrete/Patterns.h.inc"
}

void populateWithGeneratedTFHEToConcrete(mlir::RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

#endif
