// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalgOPS_H
#define ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalgOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.h"
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
/// `(tensor<...x!HLFHE.eint<$p>>, tensor<...xi$p'>) ->
/// tensor<...x!HLFHE.eint<$p>>` where `$p <= $p+1`.
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
/// `(tensor<...xi$p'>, tensor<...x!HLFHE.eint<$p>>) ->
/// tensor<...x!HLFHE.eint<$p>>` where `$p <= $p+1`.
template <typename ConcreteType>
class TensorBinaryIntEint
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorBinaryEintInt> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorBinaryIntEint(op);
  }
};

/// TensorBinary verify the operation match the following signature
/// `(tensor<...x!HLFHE.eint<$p>>, tensor<...x!HLFHE.eint<$p>>) ->
/// tensor<...x!HLFHE.eint<$p>>`
template <typename ConcreteType>
class TensorBinaryEint
    : public mlir::OpTrait::TraitBase<ConcreteType, TensorBinaryEint> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorBinaryEint(op);
  }
};

/// TensorBinary verify the operation match the following signature
/// `(tensor<...x!HLFHE.eint<$p>>) -> tensor<...x!HLFHE.eint<$p>>`
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
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h.inc"

#endif
