// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/ClientLib/CRT.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/BConcrete/Transforms/Passes.h"

namespace arith = mlir::arith;
namespace tensor = mlir::tensor;
namespace bufferization = mlir::bufferization;
namespace scf = mlir::scf;
namespace BConcrete = mlir::concretelang::BConcrete;
namespace crt = concretelang::clientlib::crt;

namespace {

char encode_crt[] = "encode_crt";

// This template rewrite pattern transforms any instance of
// `BConcreteCRTOp` operators to `BConcreteOp` on
// each block.
//
// Example:
//
// ```mlir
//  %0 = "BConcreteCRTOp"(%arg0, %arg1) {crtDecomposition = [...]}
//      : (tensor<nbBlocksxlweSizexi64>, tensor<nbBlocksxlweSizexi64>) ->
//      (tensor<nbBlocksxlweSizexi64>)
// ```
//
// becomes:
//
// ```mlir
// %c0 = arith.constant 0 : index
// %c1 = arith.constant 1 : index
// %cB = arith.constant nbBlocks : index
// %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
// %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
//   (tensor<nbBlocksxlweSizexi64>) {
//     %blockArg = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64>
//     %tmp = "BConcreteOp"(%blockArg)
//          : (tensor<lweSizexi64>) -> (tensor<lweSizexi64>)
//     %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
//     scf.yield %res : tensor<nbBlocksxlweSizexi64>
// }
// ```
template <typename BConcreteCRTOp, typename BConcreteOp>
struct BConcreteCRTUnaryOpPattern
    : public mlir::OpRewritePattern<BConcreteCRTOp> {
  BConcreteCRTUnaryOpPattern(mlir::MLIRContext *context,
                             mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<BConcreteCRTOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(BConcreteCRTOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        ((mlir::Type)op.getResult().getType()).cast<mlir::RankedTensorType>();
    auto loc = op.getLoc();
    assert(resultTy.getShape().size() == 2);
    auto shape = resultTy.getShape();

    // %c0 = arith.constant 0 : index
    // %c1 = arith.constant 1 : index
    // %cB = arith.constant nbBlocks : index
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cB = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);

    // %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), resultTy, mlir::ValueRange{});

    // %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
    //   (tensor<nbBlocksxlweSizexi64>) {
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, c0, cB, c1, init,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value i,
            mlir::ValueRange iterArgs) {
          // [%i, 0]
          mlir::SmallVector<mlir::OpFoldResult> offsets{
              i, rewriter.getI64IntegerAttr(0)};
          // [1, lweSize]
          mlir::SmallVector<mlir::OpFoldResult> sizes{
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(shape[1])};
          // [1, 1]
          mlir::SmallVector<mlir::OpFoldResult> strides{
              rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1)};

          auto blockTy = mlir::RankedTensorType::get({shape[1]},
                                                     resultTy.getElementType());

          // %blockArg = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
          //      : tensor<lweSizexi64>
          auto blockArg = builder.create<tensor::ExtractSliceOp>(
              loc, blockTy, op.ciphertext(), offsets, sizes, strides);
          // %tmp = "BConcrete.add_lwe_buffer"(%blockArg0, %blockArg1)
          //      : (tensor<lweSizexi64>, tensor<lweSizexi64>) ->
          //      (tensor<lweSizexi64>)
          auto tmp = builder.create<BConcreteOp>(loc, blockTy, blockArg);

          // %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1,
          // 1] : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
          auto res = builder.create<tensor::InsertSliceOp>(
              loc, tmp, iterArgs[0], offsets, sizes, strides);
          // scf.yield %res : tensor<nbBlocksxlweSizexi64>
          builder.create<scf::YieldOp>(loc, (mlir::Value)res);
        });

    return mlir::success();
  }
};

// This template rewrite pattern transforms any instance of
// `BConcreteCRTOp` operators to `BConcreteOp` on
// each block.
//
// Example:
//
// ```mlir
//  %0 = "BConcreteCRTOp"(%arg0, %arg1) {crtDecomposition = [...]}
//      : (tensor<nbBlocksxlweSizexi64>, tensor<nbBlocksxlweSizexi64>) ->
//      (tensor<nbBlocksxlweSizexi64>)
// ```
//
// becomes:
//
// ```mlir
// %c0 = arith.constant 0 : index
// %c1 = arith.constant 1 : index
// %cB = arith.constant nbBlocks : index
// %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
// %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
//   (tensor<nbBlocksxlweSizexi64>) {
//     %blockArg0 = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64>
//     %blockArg1 = tensor.extract_slice %arg1[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64>
//     %tmp = "BConcreteOp"(%blockArg0, %blockArg1)
//          : (tensor<lweSizexi64>, tensor<lweSizexi64>) ->
//          (tensor<lweSizexi64>)
//     %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
//     scf.yield %res : tensor<nbBlocksxlweSizexi64>
// }
// ```
template <typename BConcreteCRTOp, typename BConcreteOp>
struct BConcreteCRTBinaryOpPattern
    : public mlir::OpRewritePattern<BConcreteCRTOp> {
  BConcreteCRTBinaryOpPattern(mlir::MLIRContext *context,
                              mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<BConcreteCRTOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(BConcreteCRTOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        ((mlir::Type)op.getResult().getType()).cast<mlir::RankedTensorType>();
    auto loc = op.getLoc();
    assert(resultTy.getShape().size() == 2);
    auto shape = resultTy.getShape();

    // %c0 = arith.constant 0 : index
    // %c1 = arith.constant 1 : index
    // %cB = arith.constant nbBlocks : index
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cB = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);

    // %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), resultTy, mlir::ValueRange{});

    // %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
    //   (tensor<nbBlocksxlweSizexi64>) {
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, c0, cB, c1, init,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value i,
            mlir::ValueRange iterArgs) {
          // [%i, 0]
          mlir::SmallVector<mlir::OpFoldResult> offsets{
              i, rewriter.getI64IntegerAttr(0)};
          // [1, lweSize]
          mlir::SmallVector<mlir::OpFoldResult> sizes{
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(shape[1])};
          // [1, 1]
          mlir::SmallVector<mlir::OpFoldResult> strides{
              rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1)};

          auto blockTy = mlir::RankedTensorType::get({shape[1]},
                                                     resultTy.getElementType());

          // %blockArg0 = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
          //      : tensor<lweSizexi64>
          auto blockArg0 = builder.create<tensor::ExtractSliceOp>(
              loc, blockTy, op.lhs(), offsets, sizes, strides);
          // %blockArg1 = tensor.extract_slice %arg1[%i, 0] [1, lweSize] [1, 1]
          //      : tensor<lweSizexi64>
          auto blockArg1 = builder.create<tensor::ExtractSliceOp>(
              loc, blockTy, op.rhs(), offsets, sizes, strides);
          // %tmp = "BConcrete.add_lwe_buffer"(%blockArg0, %blockArg1)
          //      : (tensor<lweSizexi64>, tensor<lweSizexi64>) ->
          //      (tensor<lweSizexi64>)
          auto tmp =
              builder.create<BConcreteOp>(loc, blockTy, blockArg0, blockArg1);

          // %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1,
          // 1] : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
          auto res = builder.create<tensor::InsertSliceOp>(
              loc, tmp, iterArgs[0], offsets, sizes, strides);
          // scf.yield %res : tensor<nbBlocksxlweSizexi64>
          builder.create<scf::YieldOp>(loc, (mlir::Value)res);
        });

    return mlir::success();
  }
};

// This template rewrite pattern transforms any instance of
// `BConcreteCRTOp` operators to `BConcreteOp` on
// each block with the crt decomposition of the cleartext.
//
// Example:
//
// ```mlir
//  %0 = "BConcreteCRTOp"(%arg0, %x) {crtDecomposition = [d0...dn]}
//      : (tensor<nbBlocksxlweSizexi64>, i64) -> (tensor<nbBlocksxlweSizexi64>)
// ```
//
// becomes:
//
// ```mlir
// // Build the decomposition of the plaintext
// %x0_a = arith.constant 64/d0 : f64
// %x0_b = arith.mulf %x, %x0_a : i64
// %x0 = arith.fptoui %x0_b : f64 to i64
// ...
// %xn_a = arith.constant 64/dn : f64
// %xn_b = arith.mulf %x, %xn_a : i64
// %xn = arith.fptoui %xn_b : f64 to i64
// %x_decomp = tensor.from_elements %x0, ..., %xn : tensor<nbBlocksxi64>
// // Loop on blocks
// %c0 = arith.constant 0 : index
// %c1 = arith.constant 1 : index
// %cB = arith.constant nbBlocks : index
// %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
// %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
//   (tensor<nbBlocksxlweSizexi64>) {
//     %blockArg0 = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64>
//     %blockArg1 = tensor.extract %x_decomp[%i] : tensor<nbBlocksxi64>
//     %tmp = "BConcreteOp"(%blockArg0, %blockArg1)
//          : (tensor<lweSizexi64>, i64) -> (tensor<lweSizexi64>)
//     %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
//     scf.yield %res : tensor<nbBlocksxlweSizexi64>
// }
// ```
struct AddPlaintextCRTLweTensorOpPattern
    : public mlir::OpRewritePattern<BConcrete::AddPlaintextCRTLweTensorOp> {
  AddPlaintextCRTLweTensorOpPattern(mlir::MLIRContext *context,
                                    mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<BConcrete::AddPlaintextCRTLweTensorOp>(context,
                                                                      benefit) {
  }

  mlir::LogicalResult
  matchAndRewrite(BConcrete::AddPlaintextCRTLweTensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        ((mlir::Type)op.getResult().getType()).cast<mlir::RankedTensorType>();
    auto loc = op.getLoc();
    assert(resultTy.getShape().size() == 2);
    auto shape = resultTy.getShape();

    auto rhs = op.rhs();
    mlir::SmallVector<mlir::Value, 5> plaintextElements;
    uint64_t moduliProduct = 1;
    for (mlir::Attribute di : op.crtDecomposition()) {
      moduliProduct *= di.cast<mlir::IntegerAttr>().getValue().getZExtValue();
    }
    if (auto cst =
            mlir::dyn_cast_or_null<arith::ConstantIntOp>(rhs.getDefiningOp())) {
      auto apCst = cst.getValue().cast<mlir::IntegerAttr>().getValue();
      auto value = apCst.getSExtValue();

      // constant value, encode at compile time
      for (mlir::Attribute di : op.crtDecomposition()) {
        auto modulus = di.cast<mlir::IntegerAttr>().getValue().getZExtValue();

        auto encoded = crt::encode(value, modulus, moduliProduct);
        plaintextElements.push_back(
            rewriter.create<arith::ConstantIntOp>(loc, encoded, 64));
      }
    } else {
      // dynamic value, encode at runtime
      if (insertForwardDeclaration(
              op, rewriter, encode_crt,
              mlir::FunctionType::get(rewriter.getContext(),
                                      {rewriter.getI64Type(),
                                       rewriter.getI64Type(),
                                       rewriter.getI64Type()},
                                      {rewriter.getI64Type()}))
              .failed()) {
        return mlir::failure();
      }
      auto extOp =
          rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), rhs);
      auto moduliProductOp =
          rewriter.create<arith::ConstantIntOp>(loc, moduliProduct, 64);
      for (mlir::Attribute di : op.crtDecomposition()) {
        auto modulus = di.cast<mlir::IntegerAttr>().getValue().getZExtValue();
        auto modulusOp =
            rewriter.create<arith::ConstantIntOp>(loc, modulus, 64);
        plaintextElements.push_back(
            rewriter
                .create<mlir::func::CallOp>(
                    loc, encode_crt, mlir::TypeRange{rewriter.getI64Type()},
                    mlir::ValueRange{extOp, modulusOp, moduliProductOp})
                .getResult(0));
      }
    }

    // %x_decomp = tensor.from_elements %x0, ..., %xn : tensor<nbBlocksxi64>
    auto x_decomp =
        rewriter.create<tensor::FromElementsOp>(loc, plaintextElements);

    // %c0 = arith.constant 0 : index
    // %c1 = arith.constant 1 : index
    // %cB = arith.constant nbBlocks : index
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cB = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);

    // %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), resultTy, mlir::ValueRange{});

    // %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
    //   (tensor<nbBlocksxlweSizexi64>) {
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, c0, cB, c1, init,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value i,
            mlir::ValueRange iterArgs) {
          // [%i, 0]
          mlir::SmallVector<mlir::OpFoldResult> offsets{
              i, rewriter.getI64IntegerAttr(0)};
          // [1, lweSize]
          mlir::SmallVector<mlir::OpFoldResult> sizes{
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(shape[1])};
          // [1, 1]
          mlir::SmallVector<mlir::OpFoldResult> strides{
              rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1)};

          auto blockTy = mlir::RankedTensorType::get({shape[1]},
                                                     resultTy.getElementType());

          // %blockArg0 = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
          //      : tensor<lweSizexi64>
          auto blockArg0 = builder.create<tensor::ExtractSliceOp>(
              loc, blockTy, op.lhs(), offsets, sizes, strides);
          // %blockArg1 = tensor.extract %x_decomp[%i] : tensor<nbBlocksxi64>
          auto blockArg1 = builder.create<tensor::ExtractOp>(loc, x_decomp, i);
          // %tmp = "BConcreteOp"(%blockArg0, %blockArg1)
          //      : (tensor<lweSizexi64>, i64) -> (tensor<lweSizexi64>)
          auto tmp = builder.create<BConcrete::AddPlaintextLweTensorOp>(
              loc, blockTy, blockArg0, blockArg1);

          // %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1,
          // 1] : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
          auto res = builder.create<tensor::InsertSliceOp>(
              loc, tmp, iterArgs[0], offsets, sizes, strides);
          // scf.yield %res : tensor<nbBlocksxlweSizexi64>
          builder.create<scf::YieldOp>(loc, (mlir::Value)res);
        });

    return mlir::success();
  }
};

// This template rewrite pattern transforms any instance of
// `BConcreteCRTOp` operators to `BConcreteOp` on
// each block with the crt decomposition of the cleartext.
//
// Example:
//
// ```mlir
//  %0 = "BConcreteCRTOp"(%arg0, %x) {crtDecomposition = [d0...dn]}
//      : (tensor<nbBlocksxlweSizexi64>, i64) -> (tensor<nbBlocksxlweSizexi64>)
// ```
//
// becomes:
//
// ```mlir
// // Build the decomposition of the plaintext
// %x0_a = arith.constant 64/d0 : f64
// %x0_b = arith.mulf %x, %x0_a : i64
// %x0 = arith.fptoui %x0_b : f64 to i64
// ...
// %xn_a = arith.constant 64/dn : f64
// %xn_b = arith.mulf %x, %xn_a : i64
// %xn = arith.fptoui %xn_b : f64 to i64
// %x_decomp = tensor.from_elements %x0, ..., %xn : tensor<nbBlocksxi64>
// // Loop on blocks
// %c0 = arith.constant 0 : index
// %c1 = arith.constant 1 : index
// %cB = arith.constant nbBlocks : index
// %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
// %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
//   (tensor<nbBlocksxlweSizexi64>) {
//     %blockArg0 = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64>
//     %blockArg1 = tensor.extract %x_decomp[%i] : tensor<nbBlocksxi64>
//     %tmp = "BConcreteOp"(%blockArg0, %blockArg1)
//          : (tensor<lweSizexi64>, i64) -> (tensor<lweSizexi64>)
//     %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1, 1]
//          : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
//     scf.yield %res : tensor<nbBlocksxlweSizexi64>
// }
// ```
struct MulCleartextCRTLweTensorOpPattern
    : public mlir::OpRewritePattern<BConcrete::MulCleartextCRTLweTensorOp> {
  MulCleartextCRTLweTensorOpPattern(mlir::MLIRContext *context,
                                    mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<BConcrete::MulCleartextCRTLweTensorOp>(context,
                                                                      benefit) {
  }

  mlir::LogicalResult
  matchAndRewrite(BConcrete::MulCleartextCRTLweTensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        ((mlir::Type)op.getResult().getType()).cast<mlir::RankedTensorType>();
    auto loc = op.getLoc();
    assert(resultTy.getShape().size() == 2);
    auto shape = resultTy.getShape();

    // %c0 = arith.constant 0 : index
    // %c1 = arith.constant 1 : index
    // %cB = arith.constant nbBlocks : index
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cB = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);

    // %init = linalg.tensor_init [B, lweSize] : tensor<nbBlocksxlweSizexi64>
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), resultTy, mlir::ValueRange{});

    auto rhs = rewriter.create<arith::ExtUIOp>(op.getLoc(),
                                               rewriter.getI64Type(), op.rhs());

    // %0 = scf.for %i = %c0 to %cB step %c1 iter_args(%acc = %init) ->
    //   (tensor<nbBlocksxlweSizexi64>) {
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, c0, cB, c1, init,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value i,
            mlir::ValueRange iterArgs) {
          // [%i, 0]
          mlir::SmallVector<mlir::OpFoldResult> offsets{
              i, rewriter.getI64IntegerAttr(0)};
          // [1, lweSize]
          mlir::SmallVector<mlir::OpFoldResult> sizes{
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(shape[1])};
          // [1, 1]
          mlir::SmallVector<mlir::OpFoldResult> strides{
              rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1)};

          auto blockTy = mlir::RankedTensorType::get({shape[1]},
                                                     resultTy.getElementType());

          // %blockArg0 = tensor.extract_slice %arg0[%i, 0] [1, lweSize] [1, 1]
          //      : tensor<lweSizexi64>
          auto blockArg0 = builder.create<tensor::ExtractSliceOp>(
              loc, blockTy, op.lhs(), offsets, sizes, strides);

          // %tmp = BConcrete.mul_cleartext_lwe_buffer(%blockArg0, %x)
          //      : (tensor<lweSizexi64>, i64) -> (tensor<lweSizexi64>)
          auto tmp = builder.create<BConcrete::MulCleartextLweTensorOp>(
              loc, blockTy, blockArg0, rhs);

          // %res = tensor.insert_slice %tmp into %acc[%i, 0] [1, lweSize] [1,
          // 1] : tensor<lweSizexi64> into tensor<nbBlocksxlweSizexi64>
          auto res = builder.create<tensor::InsertSliceOp>(
              loc, tmp, iterArgs[0], offsets, sizes, strides);
          // scf.yield %res : tensor<nbBlocksxlweSizexi64>
          builder.create<scf::YieldOp>(loc, (mlir::Value)res);
        });

    return mlir::success();
  }
};

struct EliminateCRTOpsPass : public EliminateCRTOpsBase<EliminateCRTOpsPass> {
  void runOnOperation() final;
};

void EliminateCRTOpsPass::runOnOperation() {
  auto op = getOperation();

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());

  // add_crt_lwe_buffers
  target.addIllegalOp<BConcrete::AddCRTLweTensorOp>();
  patterns.add<BConcreteCRTBinaryOpPattern<BConcrete::AddCRTLweTensorOp,
                                           BConcrete::AddLweTensorOp>>(
      &getContext());

  // add_plaintext_crt_lwe_buffers
  target.addIllegalOp<BConcrete::AddPlaintextCRTLweTensorOp>();
  patterns.add<AddPlaintextCRTLweTensorOpPattern>(&getContext());

  // mul_cleartext_crt_lwe_buffer
  target.addIllegalOp<BConcrete::MulCleartextCRTLweTensorOp>();
  patterns.add<MulCleartextCRTLweTensorOpPattern>(&getContext());

  target.addIllegalOp<BConcrete::NegateCRTLweTensorOp>();
  patterns.add<BConcreteCRTUnaryOpPattern<BConcrete::NegateCRTLweTensorOp,
                                          BConcrete::NegateLweTensorOp>>(
      &getContext());

  // This dialect are used to transforms crt ops to bconcrete ops
  target
      .addLegalDialect<arith::ArithmeticDialect, tensor::TensorDialect,
                       scf::SCFDialect, bufferization::BufferizationDialect,
                       mlir::func::FuncDialect, BConcrete::BConcreteDialect>();

  // Apply the conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
    return;
  }
}
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<func::FuncOp>> createEliminateCRTOps() {
  return std::make_unique<EliminateCRTOpsPass>();
}
} // namespace concretelang
} // namespace mlir
