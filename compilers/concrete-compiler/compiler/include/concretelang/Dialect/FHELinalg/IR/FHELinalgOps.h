// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalgOPS_H
#define CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalgOPS_H

#include "concretelang/Dialect/FHE/IR/FHEAttrs.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir {
namespace OpTrait {

namespace impl {
LogicalResult verifyTensorBroadcastingRules(mlir::Operation *op);
LogicalResult verifyTensorBinaryEintInt(mlir::Operation *op);
LogicalResult verifyTensorBinaryIntEint(mlir::Operation *op);
LogicalResult verifyTensorBinaryEint(mlir::Operation *op);
LogicalResult verifyTensorUnaryEint(mlir::Operation *op);
} // namespace impl

/// TensorBroadcastingRules is a trait for operators that should respect the
/// broadcasting rules. All of the operands should be a RankedTensorType, the
/// result must be unique and be a RankedTensorType. The operands shape are
/// considered compatible if we compare dimensions of shapes from the right to
/// the left and if dimension are equals, or equals to one. If one of the shape
/// are smaller than the others, the missing dimension are considered to be one.
/// The result shape should have the size of the largest shape of operands and
/// each dimension `i` should be equals to the maximum of dimensions `i` of
/// each operands.
template <typename ConcreteType>
class TensorBroadcastingRules
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorBroadcastingRules> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorBroadcastingRules(op);
  }
};

/// TensorBinaryEintInt verifies that the operation matches the following
/// signature
/// `(tensor<...x!FHE.eint<$p>>, tensor<...xi$p'>) ->
/// tensor<...x!FHE.eint<$p>>` where `$p <= $p+1`.
template <typename ConcreteType>
class TensorBinaryEintInt
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorBinaryEintInt> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorBinaryEintInt(op);
  }
};

/// TensorBinaryEintInt verifies that the operation matches the following
/// signature
/// `(tensor<...xi$p'>, tensor<...x!FHE.eint<$p>>) ->
/// tensor<...x!FHE.eint<$p>>` where `$p <= $p+1`.
template <typename ConcreteType>
class TensorBinaryIntEint
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorBinaryEintInt> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorBinaryIntEint(op);
  }
};

/// TensorBinary verify the operation match the following signature
/// `(tensor<...x!FHE.eint<$p>>, tensor<...x!FHE.eint<$p>>) ->
/// tensor<...x!FHE.eint<$p>>`
template <typename ConcreteType>
class TensorBinaryEint
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorBinaryEint> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorBinaryEint(op);
  }
};

/// TensorBinary verify the operation match the following signature
/// `(tensor<...x!FHE.eint<$p>>) -> tensor<...x!FHE.eint<$p>>`
template <typename ConcreteType>
class TensorUnaryEint
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorUnaryEint> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorUnaryEint(op);
  }
};

} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h.inc"

#endif

namespace mlir {
namespace concretelang {
namespace FHELinalg {

/// Get padding from the Conv2dOp if defined, or return default value
mlir::SmallVector<int64_t, 4>
getPaddingFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp);

/// Get strides from the Conv2dOp if defined, or return default value
mlir::SmallVector<int64_t, 2>
getStridesFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp);

/// Get dilations from the Conv2dOp if defined, or return default value
mlir::SmallVector<int64_t, 2>
getDilationsFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp);

/// Get group from the Conv2dOp if defined, or return default value
int64_t getGroupFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp);

} // namespace FHELinalg
} // namespace concretelang
} // namespace mlir
