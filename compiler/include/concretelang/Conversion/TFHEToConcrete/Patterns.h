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

// This is the rewritting of the FHE::ApplyLookupTable operation, it will be
// rewritten as 3 new operations:
// - Create the required GLWE ciphertext out of the plain lookup table
// - Keyswitch the input ciphertext to match the input key of the bootstrapping
// - Bootstrap the keyswitched ciphertext with the constructed GLWE ciphertext
// Example:
// from:
// ```
// "%result = TFHE.apply_lookup_table"(% arg0, % tlu){
//   glweDimension = 1 : i32,
//   polynomialSize = 2048 : i32,
//   levelKS = 3 : i32,
//   baseLogKS = 2 : i32,
//   levelBS = 5 : i32,
//   baseLogBS = 4 : i32,
//   outputSizeKS = 600 : i32
// } : (!TFHE.glwe<{2048, 1, 64} {4}>, tensor<16xi4>)
//         ->(!TFHE.glwe<{2048, 1, 64} {4}>)
// ```
// to:
// ```
// % accumulator =
//     "Concrete.glwe_from_table"(
//         % [[TABLE]]){glweDimension = 1 : i32, p = 4 : i32, polynomialSize =
//         2048 : i32}
//     : (tensor<16xi4>)
//           ->!Concrete.glwe_ciphertext
// % keyswitched = "Concrete.keyswitch_lwe"(% arg0){
//   baseLog = 2 : i32,
//   level = 3 : i32
// } : (!Concrete.lwe_ciphertext<2048, 4>)
//         ->!Concrete.lwe_ciphertext<600, 4>
// % result = "Concrete.bootstrap_lwe"(% keyswitched, % accumulator){
//   baseLog = 4 : i32,
//   glweDimension = 1 : i32,
//   level = 5 : i32,
//   polynomialSize = 2048 : i32
// } : (!Concrete.lwe_ciphertext<600, 4>, !Concrete.glwe_ciphertext)
//         ->!Concrete.lwe_ciphertext<2048, 4>
// ```
mlir::Value createPBS(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::Value ct, mlir::Value table,
                      mlir::IntegerAttr glweDimension,
                      mlir::IntegerAttr polynomialSize,
                      mlir::IntegerAttr levelKS, mlir::IntegerAttr baseLogKS,
                      mlir::IntegerAttr levelBS, mlir::IntegerAttr baseLogBS,
                      mlir::IntegerAttr outputDimensionKS,
                      mlir::OpResult result) {
  // convert result type
  LweCiphertextType lwe_type =
      convertTypeToLWE(rewriter.getContext(), result.getType());
  // fill the the table in the GLWE accumulator
  mlir::IntegerAttr precision = rewriter.getI32IntegerAttr(lwe_type.getP());
  mlir::Value accumulator =
      rewriter
          .create<mlir::concretelang::Concrete::GlweFromTable>(
              loc, Concrete::GlweCiphertextType::get(rewriter.getContext()),
              table, polynomialSize, glweDimension, precision)
          .result();

  // keyswitch
  mlir::SmallVector<mlir::Value> ksArgs{ct};
  mlir::SmallVector<mlir::NamedAttribute> ksAttrs{
      mlir::NamedAttribute(
          mlir::Identifier::get("level", rewriter.getContext()), levelKS),
      mlir::NamedAttribute(
          mlir::Identifier::get("baseLog", rewriter.getContext()), baseLogKS),
  };
  // convert result type
  LweCiphertextType ksOutType = LweCiphertextType::get(
      rewriter.getContext(), outputDimensionKS.getInt(), precision.getInt());
  convertTypeToLWE(rewriter.getContext(), result.getType());
  mlir::Value keyswitched =
      rewriter
          .create<mlir::concretelang::Concrete::KeySwitchLweOp>(loc, ksOutType,
                                                                ksArgs, ksAttrs)
          .result();

  // bootstrap operation
  mlir::SmallVector<mlir::Value> bsArgs{keyswitched, accumulator};
  mlir::SmallVector<mlir::NamedAttribute> bsAttrs{
      mlir::NamedAttribute(
          mlir::Identifier::get("glweDimension", rewriter.getContext()),
          glweDimension),
      mlir::NamedAttribute(
          mlir::Identifier::get("polynomialSize", rewriter.getContext()),
          polynomialSize),
      mlir::NamedAttribute(
          mlir::Identifier::get("level", rewriter.getContext()), levelBS),
      mlir::NamedAttribute(
          mlir::Identifier::get("baseLog", rewriter.getContext()), baseLogBS),
  };
  mlir::Value bootstrapped =
      rewriter
          .create<mlir::concretelang::Concrete::BootstrapLweOp>(loc, lwe_type,
                                                                bsArgs, bsAttrs)
          .result();

  return bootstrapped;
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
