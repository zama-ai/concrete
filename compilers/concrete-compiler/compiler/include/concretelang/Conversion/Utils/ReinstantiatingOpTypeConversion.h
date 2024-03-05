// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_UTILS_REINSTANTIATINGOPTYPECONVERSION_H_
#define CONCRETELANG_CONVERSION_UTILS_REINSTANTIATINGOPTYPECONVERSION_H_

#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace concretelang {

// Set of types defining how attributes should be handled when
// invocating the build method of an operation upon reinstantiation
struct ReinstantiationAttributeHandling {
  // Copy attributes
  struct copy {};

  // Completely dismiss attributes by not passing a set of arguments
  // to the builder at all
  struct dismiss {};

  // Dismiss attributes by passing an empty set of arguments to the
  // builder
  struct pass_empty_vector {};
};

// Template defining how attributes should be dismissed when invoking
// the build method of an operation upon reinstantiation. In the
// default case, the argument for attributes is simply dismissed.
template <typename T> struct ReinstantiationAttributeDismissalStrategy {
  typedef ReinstantiationAttributeHandling::dismiss strategy;
};

// Template defining how attributes should be copied when invoking the
// build method of an operation upon reinstantiation. In the default
// case, the argument for attributes is forwarded to the build method.
template <typename T> struct ReinstantiationAttributeCopyStrategy {
  typedef ReinstantiationAttributeHandling::copy strategy;
};

namespace {
// Class template that defines the attribute handling strategy for
// either dismissal of attributes (if `copyAttrsSwitch` is `false`) or copying
// attributes (if `copyAttrsSwitch` is `true`).
template <typename T, bool copyAttrsSwitch> struct AttributeHandlingSwitch {};

template <typename T> struct AttributeHandlingSwitch<T, true> {
  typedef typename ReinstantiationAttributeCopyStrategy<T>::strategy strategy;
};

template <typename T> struct AttributeHandlingSwitch<T, false> {
  typedef
      typename ReinstantiationAttributeDismissalStrategy<T>::strategy strategy;
};

// Simple functor-like template invoking a rewriter with a variable
// set of arguments and an op's attributes as the last argument.
template <typename NewOpTy, typename... Args>
struct ReplaceOpWithNewOpCopyAttrs {
  static NewOpTy replace(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Operation *op, mlir::TypeRange resultTypes,
                         mlir::ValueRange operands) {
    return rewriter.replaceOpWithNewOp<NewOpTy>(op, resultTypes, operands,
                                                op->getAttrs());
  }
};

// Simple functor-like template invoking a rewriter with a variable
// set of arguments dismissing the attributes passed as the last
// argument.
template <typename NewOpTy, typename... Args>
struct ReplaceOpWithNewOpDismissAttrs {
  static NewOpTy replace(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Operation *op, mlir::TypeRange resultTypes,
                         mlir::ValueRange operands) {
    return rewriter.replaceOpWithNewOp<NewOpTy>(op, resultTypes, operands);
  }
};

// Simple functor-like template invoking a rewriter with a variable
// set of arguments dismissing the attributes by passing an empty
// set of arguments to the builder.
template <typename NewOpTy, typename... Args>
struct ReplaceOpWithNewOpEmptyAttrs {
  static NewOpTy replace(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Operation *op, mlir::TypeRange resultTypes,
                         mlir::ValueRange operands) {
    llvm::SmallVector<mlir::NamedAttribute> attrs{};
    return rewriter.replaceOpWithNewOp<NewOpTy>(op, resultTypes, operands,
                                                attrs);
  }
};

// Functor-like template that either forwards to
// `ReplaceOpWithNewOpCopyAttrs` or `ReplaceOpWithNewOpDismissAttrs`
// depending on the value of `copyAttrs`.
template <typename copyAttrsSwitch, typename OpTy, typename... Args>
struct ReplaceOpWithNewOpAttrSwitch {};

// Specialization of `ReplaceOpWithNewOpAttrSwitch` that does copy
// attributes.
template <typename OpTy, typename... Args>
struct ReplaceOpWithNewOpAttrSwitch<ReinstantiationAttributeHandling::copy,
                                    OpTy, Args...> {
  typedef ReplaceOpWithNewOpCopyAttrs<OpTy, Args...> instantiator;
};

// Specialization of `ReplaceOpWithNewOpAttrSwitch` that does NOT copy
// attributes by not passing attributes to the builder at all.
template <typename OpTy, typename... Args>
struct ReplaceOpWithNewOpAttrSwitch<ReinstantiationAttributeHandling::dismiss,
                                    OpTy, Args...> {
  typedef ReplaceOpWithNewOpDismissAttrs<OpTy, Args...> instantiator;
};

// Specialization of `ReplaceOpWithNewOpAttrSwitch` that does NOT copy
// attributes by passing an empty set of attributes to the builder.
template <typename OpTy, typename... Args>
struct ReplaceOpWithNewOpAttrSwitch<
    ReinstantiationAttributeHandling::pass_empty_vector, OpTy, Args...> {
  typedef ReplaceOpWithNewOpEmptyAttrs<OpTy, Args...> instantiator;
};

} // namespace

template <typename OldOp, typename NewOp, bool copyAttrs = false>
struct GenericOneToOneOpConversionPatternBase
    : public mlir::OpConversionPattern<OldOp> {
  GenericOneToOneOpConversionPatternBase(mlir::MLIRContext *context,
                                         mlir::TypeConverter &converter,
                                         mlir::PatternBenefit benefit = 100)
      : mlir::OpConversionPattern<OldOp>(converter, context, benefit) {}

  mlir::SmallVector<mlir::Type> convertResultTypes(OldOp oldOp) const {
    mlir::TypeConverter *converter = this->getTypeConverter();

    // Convert result types
    mlir::SmallVector<mlir::Type> resultTypes(oldOp->getNumResults());

    for (unsigned i = 0; i < oldOp->getNumResults(); i++) {
      auto result = oldOp->getResult(i);
      resultTypes[i] = converter->convertType(result.getType());
    }

    return resultTypes;
  }

  mlir::Type convertResultType(OldOp oldOp) const {
    mlir::TypeConverter *converter = this->getTypeConverter();
    return converter->convertType(oldOp->getResult(0).getType());
  }
};

// Conversion pattern that replaces an instance of an operation of the type
// `OldOp` with an instance of the type `NewOp`, taking into account operands,
// return types and possible copying attributes (iff copyAttrs is `true`).
template <typename OldOp, typename NewOp, bool copyAttrs = false>
struct GenericOneToOneOpConversionPattern
    : public GenericOneToOneOpConversionPatternBase<OldOp, NewOp, copyAttrs> {
  GenericOneToOneOpConversionPattern(mlir::MLIRContext *context,
                                     mlir::TypeConverter &converter,
                                     mlir::PatternBenefit benefit = 100)
      : GenericOneToOneOpConversionPatternBase<OldOp, NewOp, copyAttrs>(
            context, converter, benefit) {}

  virtual mlir::LogicalResult
  matchAndRewrite(OldOp oldOp,
                  typename mlir::OpConversionPattern<OldOp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Type> resultTypes = this->convertResultTypes(oldOp);

    ReplaceOpWithNewOpAttrSwitch<
        typename AttributeHandlingSwitch<NewOp, copyAttrs>::strategy,
        NewOp>::instantiator::replace(rewriter, oldOp,
                                      mlir::TypeRange{resultTypes},
                                      mlir::ValueRange{adaptor.getOperands()});

    return mlir::success();
  }
};

// Conversion pattern that retrieves the converted operands of an
// operation of the type `Op`, converts the types of the results of
// the operation and re-instantiates the operation type with the
// converted operands and result types.
template <typename Op, bool copyAttrs = false>
struct TypeConvertingReinstantiationPattern
    : public GenericOneToOneOpConversionPatternBase<Op, Op, copyAttrs> {
  TypeConvertingReinstantiationPattern(mlir::MLIRContext *context,
                                       mlir::TypeConverter &converter,
                                       mlir::PatternBenefit benefit = 100)
      : GenericOneToOneOpConversionPatternBase<Op, Op, copyAttrs>(
            context, converter, benefit) {}
  // Simple forward that makes the method specializable out of class
  // directly for this class rather than for its base
  virtual mlir::LogicalResult
  matchAndRewrite(Op op,
                  typename mlir::OpConversionPattern<Op>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Type> resultTypes = this->convertResultTypes(op);

    ReplaceOpWithNewOpAttrSwitch<
        typename AttributeHandlingSwitch<Op, copyAttrs>::strategy,
        Op>::instantiator::replace(rewriter, op, mlir::TypeRange{resultTypes},
                                   mlir::ValueRange{adaptor.getOperands()});

    return mlir::success();
  }
};

} // namespace concretelang
} // namespace mlir

#endif
