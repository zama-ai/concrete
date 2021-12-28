// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_CONVERSION_HLFHETOMIDLFHE_PATTERNS_H_
#define CONCRETELANG_CONVERSION_HLFHETOMIDLFHE_PATTERNS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOps.h"

namespace mlir {
namespace concretelang {

using HLFHE::EncryptedIntegerType;
using MidLFHE::GLWECipherTextType;

/// Converts HLFHE::EncryptedInteger into MidLFHE::GlweCiphetext
GLWECipherTextType
convertTypeEncryptedIntegerToGLWE(mlir::MLIRContext *context,
                                  EncryptedIntegerType &eint) {
  return GLWECipherTextType::get(context, -1, -1, -1, eint.getWidth());
}

mlir::Value createZeroGLWEOpFromHLFHE(mlir::PatternRewriter &rewriter,
                                      mlir::Location loc,
                                      mlir::OpResult result) {
  mlir::SmallVector<mlir::Value> args{};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto eint =
      result.getType().cast<mlir::concretelang::HLFHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  MidLFHE::ZeroGLWEOp op =
      rewriter.create<MidLFHE::ZeroGLWEOp>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createGLWEOpFromHLFHE(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value arg0,
                                  mlir::Value arg1, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 2> args{arg0, arg1};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto eint =
      result.getType().cast<mlir::concretelang::HLFHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createGLWEOpFromHLFHE(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value arg0,
                                  mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 1> args{arg0};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  auto eint =
      result.getType().cast<mlir::concretelang::HLFHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  return op.getODSResults(0).front();
}

mlir::Value
createApplyLookupTableGLWEOpFromHLFHE(mlir::PatternRewriter &rewriter,
                                      mlir::Location loc, mlir::Value arg0,
                                      mlir::Value arg1, mlir::OpResult result) {
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
      result.getType().cast<mlir::concretelang::HLFHE::EncryptedIntegerType>();
  mlir::SmallVector<mlir::Type, 1> resTypes{
      convertTypeEncryptedIntegerToGLWE(rewriter.getContext(), eint)};
  auto op = rewriter.create<concretelang::MidLFHE::ApplyLookupTable>(loc, resTypes,
                                                                 args, attrs);
  return op.getODSResults(0).front();
}

} // namespace concretelang
} // namespace mlir

namespace {
#include "concretelang/Conversion/HLFHEToMidLFHE/Patterns.h.inc"
}

void populateWithGeneratedHLFHEToMidLFHE(mlir::RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

#endif
