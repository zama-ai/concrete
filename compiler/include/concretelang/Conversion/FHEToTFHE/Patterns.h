// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_FHETOTFHE_PATTERNS_H_
#define CONCRETELANG_CONVERSION_FHETOTFHE_PATTERNS_H_

#include "concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace concretelang {

using FHE::EncryptedIntegerType;
using TFHE::GLWECipherTextType;

/// Converts FHE::EncryptedInteger into TFHE::GlweCiphetext
GLWECipherTextType
convertTypeEncryptedIntegerToGLWE(mlir::MLIRContext *context,
                                  EncryptedIntegerType eint) {
  return GLWECipherTextType::get(context, -1, -1, -1, eint.getWidth(),
                                 llvm::ArrayRef<int64_t>());
}

/// Converts the type `t` to `TFHE::GlweCiphetext` if `t` is a
/// `FHE::EncryptedInteger`, otherwise just returns `t`.
mlir::Type convertTypeToGLWEIfEncryptedIntegerType(mlir::MLIRContext *context,
                                                   mlir::Type t) {
  if (auto eint = t.dyn_cast<EncryptedIntegerType>())
    return convertTypeEncryptedIntegerToGLWE(context, eint);

  return t;
}

mlir::Value createZeroGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value> args{};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  mlir::SmallVector<mlir::Type, 1> resTypes{result.getType()};
  TFHE::ZeroGLWEOp op =
      rewriter.create<TFHE::ZeroGLWEOp>(loc, resTypes, args, attrs);
  convertOperandAndResultTypes(rewriter, op,
                               convertTypeToGLWEIfEncryptedIntegerType);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value arg0,
                                mlir::Value arg1, mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 2> args{arg0, arg1};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  mlir::SmallVector<mlir::Type, 1> resTypes{result.getType()};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  convertOperandAndResultTypes(rewriter, op,
                               convertTypeToGLWEIfEncryptedIntegerType);
  return op.getODSResults(0).front();
}

template <class Operator>
mlir::Value createGLWEOpFromFHE(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value arg0,
                                mlir::OpResult result) {
  mlir::SmallVector<mlir::Value, 1> args{arg0};
  mlir::SmallVector<mlir::NamedAttribute, 0> attrs;
  mlir::SmallVector<mlir::Type, 1> resTypes{result.getType()};
  Operator op = rewriter.create<Operator>(loc, resTypes, args, attrs);
  convertOperandAndResultTypes(rewriter, op,
                               convertTypeToGLWEIfEncryptedIntegerType);
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
