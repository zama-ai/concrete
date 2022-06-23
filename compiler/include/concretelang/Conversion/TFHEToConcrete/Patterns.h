// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_TFHETOCONCRETE_PATTERNS_H_
#define CONCRETELANG_CONVERSION_TFHETOCONCRETE_PATTERNS_H_

#include "concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    assert(glwe.getPolynomialSize() == 1);
    return LweCiphertextType::get(context, glwe.getDimension(), glwe.getP());
  }
  auto lwe = type.dyn_cast_or_null<LweCiphertextType>();
  if (lwe != nullptr) {
    return lwe;
  }
  assert(false && "expect glwe or lwe");
  return nullptr;
}

/// Converts the type `t` to an LWE type if `t` is a
/// `TFHE::GLWECipherTextType`, otherwise just returns `t`.
mlir::Type convertTypeToLWEIfTFHEType(mlir::MLIRContext *context,
                                      mlir::Type t) {
  if (auto eint = t.dyn_cast<TFHE::GLWECipherTextType>())
    return convertTypeToLWE(context, eint);

  return t;
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
  mlir::SmallVector<mlir::Type, 1> resTypes{result.getType()};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  convertOperandAndResultTypes(rewriter, op, convertTypeToLWE);

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

  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter
          .create<mlir::concretelang::Concrete::AddPlaintextLweCiphertextOp>(
              loc, result.getType(), arg0, encoded);

  convertOperandAndResultTypes(rewriter, op, convertTypeToLWEIfTFHEType);

  return op.getODSResults(0).front();
}

mlir::Value createAddPlainLweCiphertext(mlir::PatternRewriter &rewriter,
                                        mlir::Location loc, mlir::Value arg0,
                                        mlir::Value arg1,
                                        mlir::OpResult result) {
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, arg0, arg1, result,
                                             arg0.getType());
}

mlir::Value createNegLweCiphertext(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Value arg0,
                                   mlir::OpResult result) {
  auto negated =
      rewriter.create<mlir::concretelang::Concrete::NegateLweCiphertextOp>(
          loc, arg0.getType(), arg0);
  convertOperandAndResultTypes(rewriter, negated, convertTypeToLWEIfTFHEType);
  return negated.getODSResults(0).front();
}

mlir::Value createSubIntLweCiphertext(mlir::PatternRewriter &rewriter,
                                      mlir::Location loc, mlir::Value arg0,
                                      mlir::Value arg1, mlir::OpResult result) {
  auto negated_arg1 = createNegLweCiphertext(rewriter, loc, arg1, result);
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, negated_arg1, arg0,
                                             result, arg1.getType());
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

  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter
          .create<mlir::concretelang::Concrete::MulCleartextLweCiphertextOp>(
              loc, result.getType(), arg0, encoded);

  convertOperandAndResultTypes(rewriter, op, convertTypeToLWEIfTFHEType);

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
