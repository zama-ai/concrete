// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <algorithm>
#include <iostream>
#include <iterator>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/LLVM.h>

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

namespace Concrete = ::mlir::concretelang::Concrete;
namespace BConcrete = ::mlir::concretelang::BConcrete;

namespace {
struct ConcreteToBConcretePass
    : public ConcreteToBConcreteBase<ConcreteToBConcretePass> {
  void runOnOperation() final;
};
} // namespace

/// ConcreteToBConcreteTypeConverter is a TypeConverter that transform
/// `Concrete.lwe_ciphertext<dimension,p>` to `tensor<dimension+1, i64>>`
/// `tensor<...xConcrete.lwe_ciphertext<dimension,p>>` to
/// `tensor<...xdimension+1, i64>>`
class ConcreteToBConcreteTypeConverter : public mlir::TypeConverter {

public:
  ConcreteToBConcreteTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](mlir::concretelang::Concrete::PlaintextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::concretelang::Concrete::CleartextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::concretelang::Concrete::LweCiphertextType type) {
      assert(type.getDimension() != -1);
      llvm::SmallVector<int64_t, 2> shape;
      auto crt = type.getCrtDecomposition();
      if (!crt.empty()) {
        shape.push_back(crt.size());
      }
      shape.push_back(type.getDimension() + 1);
      return mlir::RankedTensorType::get(
          shape, mlir::IntegerType::get(type.getContext(), 64));
    });
    addConversion([&](mlir::concretelang::Concrete::GlweCiphertextType type) {
      assert(type.getGlweDimension() != -1);
      assert(type.getPolynomialSize() != -1);

      return mlir::RankedTensorType::get(
          {type.getPolynomialSize() * (type.getGlweDimension() + 1)},
          mlir::IntegerType::get(type.getContext(), 64));
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto lwe = type.getElementType()
                     .dyn_cast_or_null<
                         mlir::concretelang::Concrete::LweCiphertextType>();
      if (lwe == nullptr) {
        return (mlir::Type)(type);
      }
      assert(lwe.getDimension() != -1);
      mlir::SmallVector<int64_t> newShape;
      newShape.reserve(type.getShape().size() + 1);
      newShape.append(type.getShape().begin(), type.getShape().end());
      auto crt = lwe.getCrtDecomposition();
      if (!crt.empty()) {
        newShape.push_back(crt.size());
      }
      newShape.push_back(lwe.getDimension() + 1);
      mlir::Type r = mlir::RankedTensorType::get(
          newShape, mlir::IntegerType::get(type.getContext(), 64));
      return r;
    });
    addConversion([&](mlir::concretelang::RT::FutureType type) {
      return mlir::concretelang::RT::FutureType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::FutureType>()
                                .getElementType()));
    });
    addConversion([&](mlir::concretelang::RT::PointerType type) {
      return mlir::concretelang::RT::PointerType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::PointerType>()
                                .getElementType()));
    });
  }
};

template <typename ZeroOp>
struct ZeroOpPattern : public mlir::OpRewritePattern<ZeroOp> {
  ZeroOpPattern(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ZeroOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(ZeroOp zeroOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = zeroOp.getType();
    auto newResultTy = converter.convertType(resultTy);

    auto generateBody = [&](mlir::OpBuilder &nestedBuilder,
                            mlir::Location nestedLoc,
                            mlir::ValueRange blockArgs) {
      // %c0 = 0 : i64
      auto cstOp = nestedBuilder.create<mlir::arith::ConstantOp>(
          nestedLoc, nestedBuilder.getI64IntegerAttr(0));
      // tensor.yield %z : !FHE.eint<p>
      nestedBuilder.create<mlir::tensor::YieldOp>(nestedLoc, cstOp.getResult());
    };
    // tensor.generate
    rewriter.replaceOpWithNewOp<mlir::tensor::GenerateOp>(
        zeroOp, newResultTy, mlir::ValueRange{}, generateBody);

    return ::mlir::success();
  };
};

template <typename ConcreteOp, typename BConcreteOp, typename BConcreteCRTOp>
struct LowToBConcrete : public mlir::OpRewritePattern<ConcreteOp> {
  LowToBConcrete(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ConcreteOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(ConcreteOp concreteOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    mlir::TypeRange resultTyRange = concreteOp->getResultTypes();

    llvm::ArrayRef<::mlir::NamedAttribute> attributes =
        concreteOp.getOperation()->getAttrs();

    mlir::Operation *bConcreteOp;
    if (resultTyRange.size() == 1 &&
        resultTyRange.front()
            .isa<mlir::concretelang::Concrete::LweCiphertextType>()) {
      auto crt = resultTyRange.front()
                     .cast<mlir::concretelang::Concrete::LweCiphertextType>()
                     .getCrtDecomposition();
      if (crt.empty()) {
        bConcreteOp = rewriter.replaceOpWithNewOp<BConcreteOp>(
            concreteOp, resultTyRange, concreteOp.getOperation()->getOperands(),
            attributes);
      } else {
        auto newAttributes = attributes.vec();
        newAttributes.push_back(rewriter.getNamedAttr(
            "crtDecomposition", rewriter.getI64ArrayAttr(crt)));
        bConcreteOp = rewriter.replaceOpWithNewOp<BConcreteCRTOp>(
            concreteOp, resultTyRange, concreteOp.getOperation()->getOperands(),
            newAttributes);
      }
    } else {
      bConcreteOp = rewriter.replaceOpWithNewOp<BConcreteOp>(
          concreteOp, resultTyRange, concreteOp.getOperation()->getOperands(),
          attributes);
    }

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bConcreteOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct LowerKeySwitch : public mlir::OpRewritePattern<
                            mlir::concretelang::Concrete::KeySwitchLweOp> {
  LowerKeySwitch(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::Concrete::KeySwitchLweOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::KeySwitchLweOp ksOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    // construct attributes for in/out dimensions
    mlir::concretelang::Concrete::LweCiphertextType outType = ksOp.getType();
    auto outDimAttr = rewriter.getI32IntegerAttr(outType.getDimension());
    auto inputType =
        ksOp.ciphertext()
            .getType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    mlir::IntegerAttr inputDimAttr =
        rewriter.getI32IntegerAttr(inputType.getDimension());

    mlir::Operation *bKeySwitchOp = rewriter.replaceOpWithNewOp<
        mlir::concretelang::BConcrete::KeySwitchLweTensorOp>(
        ksOp, outType, ksOp.ciphertext(), ksOp.levelAttr(), ksOp.baseLogAttr(),
        inputDimAttr, outDimAttr);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bKeySwitchOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct LowerBatchedKeySwitch
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::BatchedKeySwitchLweOp> {
  LowerBatchedKeySwitch(::mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<
            mlir::concretelang::Concrete::BatchedKeySwitchLweOp>(context,
                                                                 benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::BatchedKeySwitchLweOp bksOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    mlir::concretelang::Concrete::LweCiphertextType outType =
        bksOp.getType()
            .cast<mlir::TensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();

    auto outDimAttr = rewriter.getI32IntegerAttr(outType.getDimension());
    auto inputType =
        bksOp.ciphertexts()
            .getType()
            .cast<mlir::TensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();

    mlir::IntegerAttr inputDimAttr =
        rewriter.getI32IntegerAttr(inputType.getDimension());

    mlir::Operation *bBatchedKeySwitchOp = rewriter.replaceOpWithNewOp<
        mlir::concretelang::BConcrete::BatchedKeySwitchLweTensorOp>(
        bksOp, bksOp.getType(), bksOp.ciphertexts(), bksOp.levelAttr(),
        bksOp.baseLogAttr(), inputDimAttr, outDimAttr);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bBatchedKeySwitchOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct LowerBootstrap : public mlir::OpRewritePattern<
                            mlir::concretelang::Concrete::BootstrapLweOp> {
  LowerBootstrap(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::Concrete::BootstrapLweOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::BootstrapLweOp bsOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    mlir::concretelang::Concrete::LweCiphertextType outType = bsOp.getType();
    auto inputType =
        bsOp.input_ciphertext()
            .getType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto inputDimAttr = rewriter.getI32IntegerAttr(inputType.getDimension());
    auto outputPrecisionAttr = rewriter.getI32IntegerAttr(outType.getP());
    mlir::Operation *bBootstrapOp = rewriter.replaceOpWithNewOp<
        mlir::concretelang::BConcrete::BootstrapLweTensorOp>(
        bsOp, outType, bsOp.input_ciphertext(), bsOp.lookup_table(),
        inputDimAttr, bsOp.polySizeAttr(), bsOp.levelAttr(), bsOp.baseLogAttr(),
        bsOp.glweDimensionAttr(), outputPrecisionAttr);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bBootstrapOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct LowerBatchedBootstrap
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::BatchedBootstrapLweOp> {
  LowerBatchedBootstrap(::mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<
            mlir::concretelang::Concrete::BatchedBootstrapLweOp>(context,
                                                                 benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::BatchedBootstrapLweOp bbsOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    mlir::concretelang::Concrete::LweCiphertextType outType =
        bbsOp.getType()
            .cast<mlir::TensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();

    auto inputType =
        bbsOp.input_ciphertexts()
            .getType()
            .cast<mlir::TensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();

    auto inputDimAttr = rewriter.getI32IntegerAttr(inputType.getDimension());
    auto outputPrecisionAttr = rewriter.getI32IntegerAttr(outType.getP());

    mlir::Operation *bBatchedBootstrapOp = rewriter.replaceOpWithNewOp<
        mlir::concretelang::BConcrete::BatchedBootstrapLweTensorOp>(
        bbsOp, bbsOp.getType(), bbsOp.input_ciphertexts(), bbsOp.lookup_table(),
        inputDimAttr, bbsOp.polySizeAttr(), bbsOp.levelAttr(),
        bbsOp.baseLogAttr(), bbsOp.glweDimensionAttr(), outputPrecisionAttr);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bBatchedBootstrapOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct AddPlaintextLweCiphertextOpPattern
    : public mlir::OpRewritePattern<Concrete::AddPlaintextLweCiphertextOp> {
  AddPlaintextLweCiphertextOpPattern(::mlir::MLIRContext *context,
                                     mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<Concrete::AddPlaintextLweCiphertextOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(Concrete::AddPlaintextLweCiphertextOp concreteOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto loc = concreteOp.getLoc();
    mlir::concretelang::Concrete::LweCiphertextType resultTy =
        ((mlir::Type)concreteOp->getResult(0).getType())
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    llvm::ArrayRef<::mlir::NamedAttribute> attributes =
        concreteOp.getOperation()->getAttrs();

    auto crt = resultTy.getCrtDecomposition();
    mlir::Operation *bConcreteOp;
    if (crt.empty()) {
      // Encode the plaintext value
      mlir::Value castedInt = rewriter.create<mlir::arith::ExtUIOp>(
          loc, rewriter.getIntegerType(64), concreteOp.rhs());
      mlir::Value constantShiftOp = rewriter.create<mlir::arith::ConstantOp>(
          loc,
          rewriter.getI64IntegerAttr(64 - concreteOp.getType().getP() - 1));
      auto encoded = rewriter.create<mlir::arith::ShLIOp>(
          loc, rewriter.getI64Type(), castedInt, constantShiftOp);
      bConcreteOp =
          rewriter.replaceOpWithNewOp<BConcrete::AddPlaintextLweTensorOp>(
              concreteOp, newResultTy,
              mlir::ValueRange{concreteOp.lhs(), encoded}, attributes);
    } else {
      // The encoding is done when we eliminate CRT ops
      auto newAttributes = attributes.vec();
      newAttributes.push_back(rewriter.getNamedAttr(
          "crtDecomposition", rewriter.getI64ArrayAttr(crt)));
      bConcreteOp =
          rewriter.replaceOpWithNewOp<BConcrete::AddPlaintextCRTLweTensorOp>(
              concreteOp, newResultTy, concreteOp.getOperation()->getOperands(),
              newAttributes);
    }

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bConcreteOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct MulCleartextLweCiphertextOpPattern
    : public mlir::OpRewritePattern<Concrete::MulCleartextLweCiphertextOp> {
  MulCleartextLweCiphertextOpPattern(::mlir::MLIRContext *context,
                                     mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<Concrete::MulCleartextLweCiphertextOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(Concrete::MulCleartextLweCiphertextOp concreteOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto loc = concreteOp.getLoc();
    mlir::concretelang::Concrete::LweCiphertextType resultTy =
        ((mlir::Type)concreteOp->getResult(0).getType())
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    llvm::ArrayRef<::mlir::NamedAttribute> attributes =
        concreteOp.getOperation()->getAttrs();

    auto crt = resultTy.getCrtDecomposition();
    mlir::Operation *bConcreteOp;
    if (crt.empty()) {
      // Encode the plaintext value
      mlir::Value castedInt = rewriter.create<mlir::arith::ExtUIOp>(
          loc, rewriter.getIntegerType(64), concreteOp.rhs());
      bConcreteOp =
          rewriter.replaceOpWithNewOp<BConcrete::MulCleartextLweTensorOp>(
              concreteOp, newResultTy,
              mlir::ValueRange{concreteOp.lhs(), castedInt}, attributes);
    } else {
      auto newAttributes = attributes.vec();
      newAttributes.push_back(rewriter.getNamedAttr(
          "crtDecomposition", rewriter.getI64ArrayAttr(crt)));
      bConcreteOp =
          rewriter.replaceOpWithNewOp<BConcrete::MulCleartextCRTLweTensorOp>(
              concreteOp, newResultTy, concreteOp.getOperation()->getOperands(),
              newAttributes);
    }

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bConcreteOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct ExtractSliceOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  ExtractSliceOpPattern(::mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp>(context,
                                                               benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp extractSliceOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = extractSliceOp.result().getType();
    auto lweResultTy =
        resultTy.cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto nbBlock = lweResultTy.getCrtDecomposition().size();
    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    // add 0 to the static_offsets
    mlir::SmallVector<mlir::Attribute> staticOffsets;
    staticOffsets.append(extractSliceOp.static_offsets().begin(),
                         extractSliceOp.static_offsets().end());
    if (nbBlock != 0) {
      staticOffsets.push_back(rewriter.getI64IntegerAttr(0));
    }
    staticOffsets.push_back(rewriter.getI64IntegerAttr(0));

    // add the lweSize to the sizes
    mlir::SmallVector<mlir::Attribute> staticSizes;
    staticSizes.append(extractSliceOp.static_sizes().begin(),
                       extractSliceOp.static_sizes().end());
    if (nbBlock != 0) {
      staticSizes.push_back(rewriter.getI64IntegerAttr(
          newResultTy.getDimSize(newResultTy.getRank() - 2)));
    }
    staticSizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // add 1 to the strides
    mlir::SmallVector<mlir::Attribute> staticStrides;
    staticStrides.append(extractSliceOp.static_strides().begin(),
                         extractSliceOp.static_strides().end());
    if (nbBlock != 0) {
      staticStrides.push_back(rewriter.getI64IntegerAttr(1));
    }
    staticStrides.push_back(rewriter.getI64IntegerAttr(1));

    // replace tensor.extract_slice to the new one
    mlir::tensor::ExtractSliceOp extractOp =
        rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
            extractSliceOp, newResultTy, extractSliceOp.source(),
            extractSliceOp.offsets(), extractSliceOp.sizes(),
            extractSliceOp.strides(), rewriter.getArrayAttr(staticOffsets),
            rewriter.getArrayAttr(staticSizes),
            rewriter.getArrayAttr(staticStrides));

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, extractOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

// TODO: since they are a bug on lowering extract_slice with rank reduction we
// add a linalg.tensor_collapse_shape after the extract_slice without rank
// reduction. See
// https://github.com/zama-ai/concrete-compiler-internal/issues/396.
struct ExtractOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  ExtractOpPattern(::mlir::MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::ExtractOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractOp extractOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto lweResultTy =
        extractOp.result()
            .getType()
            .dyn_cast_or_null<
                mlir::concretelang::Concrete::LweCiphertextType>();
    if (lweResultTy == nullptr) {
      return mlir::failure();
    }
    auto nbBlock = lweResultTy.getCrtDecomposition().size();
    auto newResultTy =
        converter.convertType(lweResultTy).cast<mlir::RankedTensorType>();
    auto rankOfResult = extractOp.indices().size() +
                        /* for the lwe dimension */ 1 +
                        /* for the block dimension */
                        (nbBlock == 0 ? 0 : 1);
    // [min..., 0] for static_offsets ()
    mlir::SmallVector<mlir::Attribute> staticOffsets(
        rankOfResult,
        rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
    if (nbBlock != 0) {
      staticOffsets[staticOffsets.size() - 2] = rewriter.getI64IntegerAttr(0);
    }
    staticOffsets[staticOffsets.size() - 1] = rewriter.getI64IntegerAttr(0);

    // [1..., lweDimension+1] for static_sizes or
    // [1..., nbBlock, lweDimension+1]
    mlir::SmallVector<mlir::Attribute> staticSizes(
        rankOfResult, rewriter.getI64IntegerAttr(1));
    if (nbBlock != 0) {
      staticSizes[staticSizes.size() - 2] = rewriter.getI64IntegerAttr(nbBlock);
    }
    staticSizes[staticSizes.size() - 1] = rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1));

    // [1...] for static_strides
    mlir::SmallVector<mlir::Attribute> staticStrides(
        rankOfResult, rewriter.getI64IntegerAttr(1));

    // replace tensor.extract_slice to the new one
    mlir::SmallVector<int64_t> extractedSliceShape(rankOfResult, 1);
    if (nbBlock != 0) {
      extractedSliceShape[extractedSliceShape.size() - 2] = nbBlock;
      extractedSliceShape[extractedSliceShape.size() - 1] =
          newResultTy.getDimSize(1);
    } else {
      extractedSliceShape[extractedSliceShape.size() - 1] =
          newResultTy.getDimSize(0);
    }

    auto extractedSliceType =
        mlir::RankedTensorType::get(extractedSliceShape, rewriter.getI64Type());

    auto extractedSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
        extractOp.getLoc(), extractedSliceType, extractOp.tensor(),
        extractOp.indices(), mlir::SmallVector<mlir::Value>{},
        mlir::SmallVector<mlir::Value>{}, rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));
    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, extractedSlice, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    mlir::ReassociationIndices reassociation;
    for (int64_t i = 0;
         i < extractedSliceType.getRank() - (nbBlock == 0 ? 0 : 1); i++) {
      reassociation.push_back(i);
    }

    mlir::SmallVector<mlir::ReassociationIndices> reassocs{reassociation};

    if (nbBlock != 0) {
      reassocs.push_back({extractedSliceType.getRank() - 1});
    }

    mlir::tensor::CollapseShapeOp collapseOp =
        rewriter.replaceOpWithNewOp<mlir::tensor::CollapseShapeOp>(
            extractOp, newResultTy, extractedSlice, reassocs);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, collapseOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct InsertSliceOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::InsertSliceOp> {
  InsertSliceOpPattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::InsertSliceOp>(context,
                                                              benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertSliceOp insertSliceOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = insertSliceOp.result().getType();
    auto lweResultTy =
        resultTy.cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    if (lweResultTy == nullptr) {
      return mlir::failure();
    }
    auto nbBlock = lweResultTy.getCrtDecomposition().size();
    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    // add 0 to static_offsets
    mlir::SmallVector<mlir::Attribute> staticOffsets;
    staticOffsets.append(insertSliceOp.static_offsets().begin(),
                         insertSliceOp.static_offsets().end());
    if (nbBlock != 0) {
      staticOffsets.push_back(rewriter.getI64IntegerAttr(0));
    }
    staticOffsets.push_back(rewriter.getI64IntegerAttr(0));

    // add lweDimension+1 to static_sizes
    mlir::SmallVector<mlir::Attribute> staticSizes;
    staticSizes.append(insertSliceOp.static_sizes().begin(),
                       insertSliceOp.static_sizes().end());
    if (nbBlock != 0) {
      staticSizes.push_back(rewriter.getI64IntegerAttr(
          newResultTy.getDimSize(newResultTy.getRank() - 2)));
    }
    staticSizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // add 1 to the strides
    mlir::SmallVector<mlir::Attribute> staticStrides;
    staticStrides.append(insertSliceOp.static_strides().begin(),
                         insertSliceOp.static_strides().end());
    if (nbBlock != 0) {
      staticStrides.push_back(rewriter.getI64IntegerAttr(1));
    }
    staticStrides.push_back(rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    auto newOp = rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        insertSliceOp, newResultTy, insertSliceOp.source(),
        insertSliceOp.dest(), insertSliceOp.offsets(), insertSliceOp.sizes(),
        insertSliceOp.strides(), rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, newOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

struct InsertOpPattern : public mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  InsertOpPattern(::mlir::MLIRContext *context,
                  mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::InsertOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertOp insertOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy =
        insertOp.result().getType().dyn_cast_or_null<mlir::RankedTensorType>();
    auto lweResultTy = resultTy.getElementType()
                           .dyn_cast_or_null<Concrete::LweCiphertextType>();
    if (lweResultTy == nullptr) {
      return mlir::failure();
    };
    auto hasBlock = lweResultTy.getCrtDecomposition().size() != 0;
    mlir::RankedTensorType newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    // add zeros to static_offsets
    mlir::SmallVector<mlir::OpFoldResult> offsets;
    offsets.append(insertOp.indices().begin(), insertOp.indices().end());
    offsets.push_back(rewriter.getIndexAttr(0));
    if (hasBlock) {
      offsets.push_back(rewriter.getIndexAttr(0));
    }

    // Inserting a smaller tensor into a (potentially) bigger one. Set
    // dimensions for all leading dimensions of the target tensor not
    // present in the source to 1.
    mlir::SmallVector<mlir::OpFoldResult> sizes(insertOp.indices().size(),
                                                rewriter.getI64IntegerAttr(1));

    // Add size for the bufferized source element
    if (hasBlock) {
      sizes.push_back(rewriter.getI64IntegerAttr(
          newResultTy.getDimSize(newResultTy.getRank() - 2)));
    }
    sizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // Set stride of all dimensions to 1
    mlir::SmallVector<mlir::OpFoldResult> strides(
        newResultTy.getRank(), rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    mlir::tensor::InsertSliceOp insertSliceOp =
        rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
            insertOp, insertOp.getOperand(0), insertOp.dest(), offsets, sizes,
            strides);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, insertSliceOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// FromElementsOpPatterns transform each tensor.from_elements that operates on
/// Concrete.lwe_ciphertext
///
/// refs: check_tests/Conversion/ConcreteToBConcrete/tensor_from_elements.mlir
struct FromElementsOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::FromElementsOp> {
  FromElementsOpPattern(::mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::FromElementsOp>(context,
                                                               benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::FromElementsOp fromElementsOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    auto resultTy = fromElementsOp.result().getType();
    if (converter.isLegal(resultTy)) {
      return mlir::failure();
    }
    auto oldTensorResultTy = resultTy.cast<mlir::RankedTensorType>();
    auto oldRank = oldTensorResultTy.getRank();

    auto newTensorResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();
    auto newRank = newTensorResultTy.getRank();
    auto newShape = newTensorResultTy.getShape();

    mlir::Value tensor = rewriter.create<mlir::bufferization::AllocTensorOp>(
        fromElementsOp.getLoc(), newTensorResultTy, mlir::ValueRange{});

    // sizes are [1, ..., 1, diffShape...]
    llvm::SmallVector<mlir::OpFoldResult> sizes(oldRank,
                                                rewriter.getI64IntegerAttr(1));
    for (auto i = newRank - oldRank; i > 0; i--) {
      sizes.push_back(rewriter.getI64IntegerAttr(*(newShape.end() - i)));
    }

    // strides are [1, ..., 1]
    llvm::SmallVector<mlir::OpFoldResult> oneStrides(
        newShape.size(), rewriter.getI64IntegerAttr(1));

    // start with offets [0, ..., 0]
    llvm::SmallVector<int64_t> currentOffsets(newRank, 0);

    // for each elements insert_slice with right offet
    for (auto elt : llvm::enumerate(fromElementsOp.elements())) {
      // Just create offsets as attributes
      llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
      offsets.reserve(currentOffsets.size());
      std::transform(currentOffsets.begin(), currentOffsets.end(),
                     std::back_inserter(offsets),
                     [&](auto v) { return rewriter.getI64IntegerAttr(v); });
      mlir::tensor::InsertSliceOp insOp =
          rewriter.create<mlir::tensor::InsertSliceOp>(
              fromElementsOp.getLoc(),
              /* src: */ elt.value(),
              /* dst: */ tensor,
              /* offs: */ offsets,
              /* sizes: */ sizes,
              /* strides: */ oneStrides);

      mlir::concretelang::convertOperandAndResultTypes(
          rewriter, insOp, [&](mlir::MLIRContext *, mlir::Type t) {
            return converter.convertType(t);
          });

      tensor = insOp.getResult();

      // Increment the offsets
      for (auto i = newRank - 2; i >= 0; i--) {
        if (currentOffsets[i] == newShape[i] - 1) {
          currentOffsets[i] = 0;
          continue;
        }
        currentOffsets[i]++;
        break;
      }
    }

    rewriter.replaceOp(fromElementsOp, tensor);
    return ::mlir::success();
  };
};

// This template rewrite pattern transforms any instance of
// `ShapeOp` operators that operates on tensor of lwe ciphertext by adding the
// lwe size as a size of the tensor result and by adding a trivial
// reassociation at the end of the reassociations map.
//
// Example:
//
// ```mlir
// %0 = "ShapeOp" %arg0 [reassocations...]
//        : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> into
//          tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
// ```
//
// becomes:
//
// ```mlir
// %0 = "ShapeOp" %arg0 [reassociations..., [inRank or outRank]]
//        : tensor<...xlweDimesion+1xi64> into
//          tensor<...xlweDimesion+1xi64>
// ```
template <typename ShapeOp, typename VecTy, bool inRank>
struct TensorShapeOpPattern : public mlir::OpRewritePattern<ShapeOp> {
  TensorShapeOpPattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ShapeOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(ShapeOp shapeOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = ((mlir::Type)shapeOp.result().getType()).cast<VecTy>();
    auto lweResultTy =
        ((mlir::Type)resultTy.getElementType())
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();

    auto newResultTy =
        ((mlir::Type)converter.convertType(resultTy)).cast<VecTy>();

    auto reassocTy =
        ((mlir::Type)converter.convertType(
             (inRank ? shapeOp.src() : shapeOp.result()).getType()))
            .cast<VecTy>();

    auto oldReassocs = shapeOp.getReassociationIndices();
    mlir::SmallVector<mlir::ReassociationIndices> newReassocs;
    newReassocs.append(oldReassocs.begin(), oldReassocs.end());
    // add [rank-1] to reassociations if crt decomp
    if (!lweResultTy.getCrtDecomposition().empty()) {
      mlir::ReassociationIndices lweAssoc;
      lweAssoc.push_back(reassocTy.getRank() - 2);
      newReassocs.push_back(lweAssoc);
    }

    // add [rank] to reassociations
    {
      mlir::ReassociationIndices lweAssoc;
      lweAssoc.push_back(reassocTy.getRank() - 1);
      newReassocs.push_back(lweAssoc);
    }

    ShapeOp op = rewriter.replaceOpWithNewOp<ShapeOp>(
        shapeOp, newResultTy, shapeOp.src(), newReassocs);

    // fix operand types
    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, op, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// Add the instantiated TensorShapeOpPattern rewrite pattern with the `ShapeOp`
/// to the patterns set and populate the conversion target.
template <typename ShapeOp, typename VecTy, bool inRank>
void insertTensorShapeOpPattern(mlir::MLIRContext &context,
                                mlir::RewritePatternSet &patterns,
                                mlir::ConversionTarget &target) {
  patterns.insert<TensorShapeOpPattern<ShapeOp, VecTy, inRank>>(&context);
  target.addDynamicallyLegalOp<ShapeOp>([&](mlir::Operation *op) {
    ConcreteToBConcreteTypeConverter converter;
    return converter.isLegal(op->getResultTypes()) &&
           converter.isLegal(op->getOperandTypes());
  });
}

struct AllocTensorOpPattern
    : public mlir::OpRewritePattern<mlir::bufferization::AllocTensorOp> {
  AllocTensorOpPattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::bufferization::AllocTensorOp>(context,
                                                                     benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::AllocTensorOp allocTensorOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    mlir::RankedTensorType resultTy =
        allocTensorOp.getType().dyn_cast<mlir::RankedTensorType>();

    if (!resultTy || !resultTy.hasStaticShape())
      return mlir::failure();

    mlir::RankedTensorType newResultTy =
        converter.convertType(resultTy).dyn_cast<mlir::RankedTensorType>();

    if (resultTy.getShape().size() != newResultTy.getShape().size()) {
      rewriter.replaceOpWithNewOp<mlir::bufferization::AllocTensorOp>(
          allocTensorOp, newResultTy, mlir::ValueRange{});
    }

    return ::mlir::success();
  };
};

struct ForOpPattern : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  ForOpPattern(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::scf::ForOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp forOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    // TODO: Check if there is a cleaner way to modify the types in
    // place through appropriate interfaces or by reconstructing the
    // ForOp with the right types.
    rewriter.updateRootInPlace(forOp, [&] {
      for (mlir::Value initArg : forOp.getInitArgs()) {
        mlir::Type convertedType = converter.convertType(initArg.getType());
        initArg.setType(convertedType);
      }

      for (mlir::Value &blockArg : forOp.getBody()->getArguments()) {
        mlir::Type convertedType = converter.convertType(blockArg.getType());
        blockArg.setType(convertedType);
      }

      for (mlir::OpResult result : forOp.getResults()) {
        mlir::Type convertedType = converter.convertType(result.getType());
        result.setType(convertedType);
      }
    });

    return ::mlir::success();
  };
};

void ConcreteToBConcretePass::runOnOperation() {
  auto op = this->getOperation();

  // Then convert ciphertext to tensor or add a dimension to tensor of
  // ciphertext and memref of ciphertext
  {
    mlir::ConversionTarget target(getContext());
    ConcreteToBConcreteTypeConverter converter;
    mlir::RewritePatternSet patterns(&getContext());

    // All BConcrete ops are legal after the conversion
    target.addLegalDialect<mlir::concretelang::BConcrete::BConcreteDialect>();

    // Add Concrete ops are illegal after the conversion
    target.addIllegalDialect<mlir::concretelang::Concrete::ConcreteDialect>();

    target.addLegalDialect<mlir::arith::ArithmeticDialect>();

    // Add patterns to convert the zero ops to tensor.generate
    patterns
        .insert<ZeroOpPattern<mlir::concretelang::Concrete::ZeroTensorLWEOp>,
                ZeroOpPattern<mlir::concretelang::Concrete::ZeroLWEOp>>(
            &getContext());
    target.addLegalOp<mlir::tensor::GenerateOp, mlir::tensor::YieldOp>();

    // Add patterns to trivialy convert Concrete op to the equivalent
    // BConcrete op
    patterns.insert<
        LowerBootstrap, LowerBatchedBootstrap, LowerKeySwitch,
        LowerBatchedKeySwitch,
        LowToBConcrete<mlir::concretelang::Concrete::AddLweCiphertextsOp,
                       mlir::concretelang::BConcrete::AddLweTensorOp,
                       BConcrete::AddCRTLweTensorOp>,
        AddPlaintextLweCiphertextOpPattern, MulCleartextLweCiphertextOpPattern,
        LowToBConcrete<mlir::concretelang::Concrete::NegateLweCiphertextOp,
                       mlir::concretelang::BConcrete::NegateLweTensorOp,
                       BConcrete::NegateCRTLweTensorOp>,
        LowToBConcrete<Concrete::WopPBSLweOp, BConcrete::WopPBSCRTLweTensorOp,
                       BConcrete::WopPBSCRTLweTensorOp>>(&getContext());

    // Add patterns to rewrite tensor operators that works on encrypted
    // tensors
    patterns
        .insert<ExtractSliceOpPattern, ExtractOpPattern, InsertSliceOpPattern,
                InsertOpPattern, FromElementsOpPattern>(&getContext());

    target.addDynamicallyLegalOp<mlir::tensor::ExtractSliceOp,
                                 mlir::tensor::ExtractOp, mlir::scf::YieldOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResultTypes()) &&
                 converter.isLegal(op->getOperandTypes());
        });

    patterns.insert<AllocTensorOpPattern>(&getContext());

    target.addDynamicallyLegalOp<mlir::tensor::InsertSliceOp,
                                 mlir::tensor::FromElementsOp,
                                 mlir::bufferization::AllocTensorOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResult(0).getType());
        });
    target.addLegalOp<mlir::memref::CopyOp>();

    patterns.insert<ForOpPattern>(&getContext());

    // Add patterns to rewrite some of memref ops that was introduced by the
    // linalg bufferization of encrypted tensor (first conversion of this
    // pass)
    insertTensorShapeOpPattern<mlir::memref::ExpandShapeOp, mlir::MemRefType,
                               false>(getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::tensor::ExpandShapeOp, mlir::TensorType,
                               false>(getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::memref::CollapseShapeOp, mlir::MemRefType,
                               true>(getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::tensor::CollapseShapeOp, mlir::TensorType,
                               true>(getContext(), patterns, target);

    target.addDynamicallyLegalOp<
        mlir::arith::ConstantOp, mlir::scf::ForOp, mlir::scf::ParallelOp,
        mlir::scf::YieldOp, mlir::AffineApplyOp, mlir::memref::SubViewOp,
        mlir::memref::LoadOp, mlir::memref::TensorStoreOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResultTypes()) &&
                 converter.isLegal(op->getOperandTypes());
        });

    // Add patterns to do the conversion of func
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return converter.isSignatureLegal(funcOp.getFunctionType()) &&
                 converter.isLegal(&funcOp.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
        [&](mlir::func::ConstantOp op) {
          return FunctionConstantOpConversion<
              ConcreteToBConcreteTypeConverter>::isLegal(op, converter);
        });
    patterns
        .insert<FunctionConstantOpConversion<ConcreteToBConcreteTypeConverter>>(
            &getContext(), converter);

    target.addDynamicallyLegalOp<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
      return converter.isLegal(forOp.getInitArgs().getTypes()) &&
             converter.isLegal(forOp.getResults().getTypes());
    });

    // Add pattern for return op
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResultTypes()) &&
                 converter.isLegal(op->getOperandTypes());
        });

    // Conversion of RT Dialect Ops
    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<mlir::scf::YieldOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::MakeReadyFutureOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::AwaitFutureOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::CreateAsyncTaskOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::WorkFunctionReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::RegisterTaskWorkFunctionOp>>(&getContext(),
                                                                 converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::MakeReadyFutureOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::AwaitFutureOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::CreateAsyncTaskOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
        target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::WorkFunctionReturnOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::RegisterTaskWorkFunctionOp>(target, converter);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertConcreteToBConcretePass() {
  return std::make_unique<ConcreteToBConcretePass>();
}
} // namespace concretelang
} // namespace mlir
