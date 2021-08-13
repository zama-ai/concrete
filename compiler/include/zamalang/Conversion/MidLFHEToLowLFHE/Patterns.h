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

LweCiphertextType convertTypeGLWEToLWE(mlir::MLIRContext *context,
                                       GLWECipherTextType &glwe) {
  return LweCiphertextType::get(context);
}

PlaintextType convertPlaintextTypeFromGlwe(mlir::MLIRContext *context,
                                           GLWECipherTextType &type) {
  return PlaintextType::get(context, type.getP() + 1);
}

CleartextType convertCleartextTypeFromGlwe(mlir::MLIRContext *context,
                                           GLWECipherTextType &type) {
  return CleartextType::get(context, type.getP() + 1);
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
      convertTypeGLWEToLWE(rewriter.getContext(), glwe)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

mlir::Value createAddPlainLweCiphertextWithGlwe(
    mlir::PatternRewriter rewriter, mlir::Location loc, mlir::Value arg0,
    mlir::Value arg1, mlir::OpResult result, GLWECipherTextType glwe) {
  PlaintextType encoded_type =
      convertPlaintextTypeFromGlwe(rewriter.getContext(), glwe);
  // encode int into plaintext
  mlir::Value encoded =
      rewriter
          .create<mlir::zamalang::LowLFHE::EncodeIntOp>(loc, encoded_type, arg1)
          .plaintext();
  // convert result type
  GLWECipherTextType glwe_type = result.getType().cast<GLWECipherTextType>();
  LweCiphertextType lwe_type =
      convertTypeGLWEToLWE(rewriter.getContext(), glwe_type);
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
  auto arg1_type = arg1.getType().cast<GLWECipherTextType>();
  auto negated_arg1 =
      rewriter
          .create<mlir::zamalang::LowLFHE::NegateLweCiphertextOp>(
              loc, convertTypeGLWEToLWE(rewriter.getContext(), arg1_type), arg1)
          .result();
  return createAddPlainLweCiphertextWithGlwe(rewriter, loc, negated_arg1, arg0,
                                             result, arg1_type);
}

mlir::Value createMulClearLweCiphertext(mlir::PatternRewriter rewriter,
                                        mlir::Location loc, mlir::Value arg0,
                                        mlir::Value arg1,
                                        mlir::OpResult result) {
  auto glwe = arg0.getType().cast<GLWECipherTextType>();
  CleartextType encoded_type =
      convertCleartextTypeFromGlwe(rewriter.getContext(), glwe);
  // encode int into plaintext
  mlir::Value encoded = rewriter
                            .create<mlir::zamalang::LowLFHE::IntToCleartextOp>(
                                loc, encoded_type, arg1)
                            .cleartext();
  // convert result type
  GLWECipherTextType glwe_type = result.getType().cast<GLWECipherTextType>();
  LweCiphertextType lwe_type =
      convertTypeGLWEToLWE(rewriter.getContext(), glwe_type);
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
