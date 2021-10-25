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

mlir::Value createZeroLWEOpFromMidLFHE(mlir::PatternRewriter &rewriter,
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
mlir::Value createLowLFHEOpFromMidLFHE(mlir::PatternRewriter &rewriter,
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
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value arg0,
    mlir::Value arg1, mlir::OpResult result, mlir::Type encryptedType) {
  PlaintextType encoded_type =
      convertPlaintextTypeFromType(rewriter.getContext(), encryptedType);
  // encode int into plaintext
  mlir::Value encoded =
      rewriter
          .create<mlir::zamalang::LowLFHE::EncodeIntOp>(loc, encoded_type, arg1)
          .plaintext();
  // convert result type
  LweCiphertextType lwe_type =
      convertTypeToLWE(rewriter.getContext(), result.getType());
  // replace op using the encoded plaintext instead of int
  auto op =
      rewriter.create<mlir::zamalang::LowLFHE::AddPlaintextLweCiphertextOp>(
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
          .create<mlir::zamalang::LowLFHE::NegateLweCiphertextOp>(
              loc, convertTypeToLWE(rewriter.getContext(), arg1_type), arg1)
          .result();
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, negated_arg1, arg0,
                                             result, arg1_type);
}

mlir::Value createMulClearLweCiphertext(mlir::PatternRewriter &rewriter,
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

// This is the rewritting of the HLFHE::ApplyLookupTable operation, it will be
// rewritten as 3 new operations:
// - Create the required GLWE ciphertext out of the plain lookup table
// - Keyswitch the input ciphertext to match the input key of the bootstrapping
// - Bootstrap the keyswitched ciphertext with the constructed GLWE ciphertext
// Example:
// from:
// ```
// "%result = MidLFHE.apply_lookup_table"(% arg0, % tlu){
//   k = 1 : i32,
//   polynomialSize = 2048 : i32,
//   levelKS = 3 : i32,
//   baseLogKS = 2 : i32,
//   levelBS = 5 : i32,
//   baseLogBS = 4 : i32,
//   outputSizeKS = 600 : i32
// } : (!MidLFHE.glwe<{2048, 1, 64} {4}>, tensor<16xi4>)
//         ->(!MidLFHE.glwe<{2048, 1, 64} {4}>)
// ```
// to:
// ```
// % accumulator =
//     "LowLFHE.glwe_from_table"(
//         % [[TABLE]]){k = 1 : i32, p = 4 : i32, polynomialSize = 2048 : i32}
//     : (tensor<16xi4>)
//           ->!LowLFHE.glwe_ciphertext
// % keyswitched = "LowLFHE.keyswitch_lwe"(% arg0){
//   baseLog = 2 : i32,
//   inputLweSize = 1 : i32,
//   level = 3 : i32,
//   outputLweSize = 600 : i32
// } : (!LowLFHE.lwe_ciphertext<2048, 4>)
//         ->!LowLFHE.lwe_ciphertext<600, 4>
// % result = "LowLFHE.bootstrap_lwe"(% keyswitched, % accumulator){
//   baseLog = 4 : i32,
//   k = 1 : i32,
//   level = 5 : i32,
//   polynomialSize = 2048 : i32
// } : (!LowLFHE.lwe_ciphertext<600, 4>, !LowLFHE.glwe_ciphertext)
//         ->!LowLFHE.lwe_ciphertext<2048, 4>
// ```
mlir::Value createPBS(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::Value ct, mlir::Value table, mlir::IntegerAttr k,
                      mlir::IntegerAttr polynomialSize,
                      mlir::IntegerAttr levelKS, mlir::IntegerAttr baseLogKS,
                      mlir::IntegerAttr levelBS, mlir::IntegerAttr baseLogBS,
                      mlir::IntegerAttr outputSizeKS, mlir::OpResult result) {
  // convert result type
  GLWECipherTextType glwe_type = result.getType().cast<GLWECipherTextType>();
  LweCiphertextType lwe_type =
      convertTypeToLWE(rewriter.getContext(), glwe_type);
  // fill the the table in the GLWE accumulator
  mlir::IntegerAttr precision = rewriter.getI32IntegerAttr(glwe_type.getP());
  mlir::Value accumulator =
      rewriter
          .create<mlir::zamalang::LowLFHE::GlweFromTable>(
              loc, LowLFHE::GlweCiphertextType::get(rewriter.getContext()),
              table, polynomialSize, k, precision)
          .result();

  // keyswitch
  auto ct_type = ct.getType().cast<GLWECipherTextType>();
  mlir::SmallVector<mlir::Value> ksArgs{ct};
  mlir::SmallVector<mlir::NamedAttribute> ksAttrs{
      mlir::NamedAttribute(
          mlir::Identifier::get("inputLweSize", rewriter.getContext()), k),
      mlir::NamedAttribute(
          mlir::Identifier::get("outputLweSize", rewriter.getContext()),
          outputSizeKS),
      mlir::NamedAttribute(
          mlir::Identifier::get("level", rewriter.getContext()), levelKS),
      mlir::NamedAttribute(
          mlir::Identifier::get("baseLog", rewriter.getContext()), baseLogKS),
  };
  auto ksOutType = LweCiphertextType::get(
      rewriter.getContext(), outputSizeKS.getInt(), ct_type.getP());
  mlir::Value keyswitched =
      rewriter
          .create<mlir::zamalang::LowLFHE::KeySwitchLweOp>(loc, ksOutType,
                                                           ksArgs, ksAttrs)
          .result();

  // bootstrap operation
  mlir::SmallVector<mlir::Value> bsArgs{keyswitched, accumulator};
  mlir::SmallVector<mlir::NamedAttribute> bsAttrs{
      mlir::NamedAttribute(mlir::Identifier::get("k", rewriter.getContext()),
                           k),
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
          .create<mlir::zamalang::LowLFHE::BootstrapLweOp>(loc, lwe_type,
                                                           bsArgs, bsAttrs)
          .result();

  return bootstrapped;
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
