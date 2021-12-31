// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_CONVERSION_FHETOTFHE_PATTERNS_H_
#define CONCRETELANG_CONVERSION_FHETOTFHE_PATTERNS_H_

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace concretelang {

using FHE::EncryptedIntegerType;
using TFHE::GLWECipherTextType;

/// Converts FHE::EncryptedInteger into TFHE::GlweCiphetext
GLWECipherTextType
convertTypeEncryptedIntegerToGLWE(mlir::MLIRContext *context,
                                  EncryptedIntegerType &eint) {
  return GLWECipherTextType::get(context, -1, -1, -1, eint.getWidth());
}

mlir::Value createZeroGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value> args{};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto eint =
      result.getType().cast<mlir::concretelang::FHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  TFHE::ZeroGLWEOp op =
      rewriter.create<TFHE::ZeroGLWEOp>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value arg0,
                                mlir::Value arg1, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 2> args{arg0, arg1};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto eint =
      result.getType().cast<mlir::concretelang::FHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value arg0,
                                mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 1> args{arg0};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto eint =
      result.getType().cast<mlir::concretelang::FHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

mlir::Value createApplyLookupTableGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                                mlir::Location loc,
                                                mlir::Value arg0,
                                                mlir::Value arg1,
                                                mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 2> args{arg0, arg1};

  auto context = rewriter.getContext();
  auto unset = mlir::IntegerAttr::get(IntegerType::get(context, 32), -1);
  mlir::SmallVector<mlir::NamedAttribute, 6> attrs{
      mlir::NamedAttribute(mlir::Identifier::get("glweDimension", context),
                           unset),
      mlir::NamedAttribute(mlir::Identifier::get("polynomialSize", context),
                           unset),
      mlir::NamedAttribute(mlir::Identifier::get("levelKS", context), unset),
      mlir::NamedAttribute(mlir::Identifier::get("baseLogKS", context), unset),
      mlir::NamedAttribute(mlir::Identifier::get("levelBS", context), unset),
      mlir::NamedAttribute(mlir::Identifier::get("baseLogBS", context), unset),
      mlir::NamedAttribute(mlir::Identifier::get("outputSizeKS", context),
                           unset),
  };
  auto eint =
      result.getType().cast<mlir::concretelang::FHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  auto op = rewriter.create<concretelang::TFHE::ApplyLookupTable>(loc, resTypes,
                                                                  args, attrs);
  return op.getODSResults(0).front();
}

} // namespace concretelang
} // namespace mlir

namespace {
#include "concretelang/Conversion/FHEToTFHE/Patterns.h.inc"
}

void populateWithGeneratedFHEToTFHE(mlir::RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

#endif
