// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <unordered_set>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.h"

namespace mlir {
namespace OpTrait {
namespace impl {

LogicalResult verifyTensorBroadcastingRules(
    mlir::Operation *op, llvm::SmallVector<mlir::RankedTensorType> operands,
    mlir::RankedTensorType result) {
  llvm::SmallVector<llvm::ArrayRef<int64_t>> operandsShapes;
  size_t maxOperandsDim = 0;
  auto resultShape = result.getShape();
  for (size_t i = 0; i < operands.size(); i++) {
    auto shape = operands[i].getShape();
    operandsShapes.push_back(shape);
    maxOperandsDim = std::max(shape.size(), maxOperandsDim);
  }
  // Check the result has the same number of dimension than the highest
  // dimension of operands
  if (resultShape.size() != maxOperandsDim) {
    op->emitOpError()
        << "should have the number of dimensions of the result equal to the "
           "highest number of dimensions of operands"
        << ", got " << result.getShape().size() << " expect " << maxOperandsDim;
    return mlir::failure();
  }

  // For all dimension
  for (size_t i = 0; i < maxOperandsDim; i++) {
    int64_t expectedResultDim = 1;

    // Check the dimension of operands shape are compatible, i.e. equals or 1
    for (size_t j = 0; j < operandsShapes.size(); j++) {
      if (i < maxOperandsDim - operandsShapes[j].size()) {
        continue;
      }
      auto k = i - (maxOperandsDim - operandsShapes[j].size());
      auto operandDim = operandsShapes[j][k];
      if (expectedResultDim != 1 && operandDim != 1 &&
          operandDim != expectedResultDim) {
        op->emitOpError() << "has the dimension #"
                          << (operandsShapes[j].size() - k)
                          << " of the operand #" << j
                          << " incompatible with other operands"
                          << ", got " << operandDim << " expect 1 or "
                          << expectedResultDim;
        return mlir::failure();
      }

      expectedResultDim = std::max(operandDim, expectedResultDim);
    }

    // Check the dimension of the result is compatible with dimesion of the
    // operands
    if (resultShape[i] != expectedResultDim) {
      op->emitOpError() << "has the dimension #" << (maxOperandsDim - i)
                        << " of the result incompatible with operands dimension"
                        << ", got " << resultShape[i] << " expect "
                        << expectedResultDim;
      return mlir::failure();
    }
  }

  return mlir::success();
}

LogicalResult verifyTensorBroadcastingRules(mlir::Operation *op) {
  // Check operands type are ranked tensor
  llvm::SmallVector<mlir::RankedTensorType> tensorOperands;
  unsigned i = 0;
  for (auto opType : op->getOperandTypes()) {
    auto tensorType = opType.dyn_cast_or_null<mlir::RankedTensorType>();
    if (tensorType == nullptr) {
      op->emitOpError() << " should have a ranked tensor as operand #" << i;
      return mlir::failure();
    }
    tensorOperands.push_back(tensorType);
    i++;
  }
  // Check number of result is 1
  if (op->getNumResults() != 1) {
    op->emitOpError() << "should have exactly 1 result, got "
                      << op->getNumResults();
  }
  auto tensorResult =
      op->getResult(0).getType().dyn_cast_or_null<mlir::RankedTensorType>();
  if (tensorResult == nullptr) {
    op->emitOpError(llvm::Twine("should have a ranked tensor as result"));
    return mlir::failure();
  }
  return verifyTensorBroadcastingRules(op, tensorOperands, tensorResult);
}

LogicalResult verifyTensorBinaryEintInt(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    op->emitOpError() << "should have exactly 2 operands";
    return mlir::failure();
  }
  auto op0Ty = op->getOperand(0).getType().dyn_cast_or_null<mlir::TensorType>();
  auto op1Ty = op->getOperand(1).getType().dyn_cast_or_null<mlir::TensorType>();
  if (op0Ty == nullptr || op1Ty == nullptr) {
    op->emitOpError() << "should have both operands as tensor";
    return mlir::failure();
  }
  auto el0Ty =
      op0Ty.getElementType()
          .dyn_cast_or_null<mlir::concretelang::FHE::EncryptedIntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have a !FHE.eint as the element type of the "
                         "tensor of operand #0";
    return mlir::failure();
  }
  auto el1Ty = op1Ty.getElementType().dyn_cast_or_null<mlir::IntegerType>();
  if (el1Ty == nullptr) {
    op->emitOpError() << "should have an integer as the element type of the "
                         "tensor of operand #1";
    return mlir::failure();
  }
  if (el1Ty.getWidth() > el0Ty.getWidth() + 1) {
    op->emitOpError()
        << "should have the width of integer values less or equals "
           "than the width of encrypted values + 1";
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult verifyTensorBinaryIntEint(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    op->emitOpError() << "should have exactly 2 operands";
    return mlir::failure();
  }
  auto op0Ty = op->getOperand(0).getType().dyn_cast_or_null<mlir::TensorType>();
  auto op1Ty = op->getOperand(1).getType().dyn_cast_or_null<mlir::TensorType>();
  if (op0Ty == nullptr || op1Ty == nullptr) {
    op->emitOpError() << "should have both operands as tensor";
    return mlir::failure();
  }
  auto el0Ty = op0Ty.getElementType().dyn_cast_or_null<mlir::IntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have an integer as the element type of the "
                         "tensor of operand #0";
    return mlir::failure();
  }
  auto el1Ty =
      op1Ty.getElementType()
          .dyn_cast_or_null<mlir::concretelang::FHE::EncryptedIntegerType>();
  if (el1Ty == nullptr) {
    op->emitOpError() << "should have a !FHE.eint as the element type of the "
                         "tensor of operand #1";
    return mlir::failure();
  }
  if (el1Ty.getWidth() > el0Ty.getWidth() + 1) {
    op->emitOpError()
        << "should have the width of integer values less or equals "
           "than the width of encrypted values + 1";
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult verifyTensorBinaryEint(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    op->emitOpError() << "should have exactly 2 operands";
    return mlir::failure();
  }
  auto op0Ty = op->getOperand(0).getType().dyn_cast_or_null<mlir::TensorType>();
  auto op1Ty = op->getOperand(1).getType().dyn_cast_or_null<mlir::TensorType>();
  if (op0Ty == nullptr || op1Ty == nullptr) {
    op->emitOpError() << "should have both operands as tensor";
    return mlir::failure();
  }
  auto el0Ty =
      op0Ty.getElementType()
          .dyn_cast_or_null<mlir::concretelang::FHE::EncryptedIntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have a !FHE.eint as the element type of the "
                         "tensor of operand #0";
    return mlir::failure();
  }
  auto el1Ty =
      op1Ty.getElementType()
          .dyn_cast_or_null<mlir::concretelang::FHE::EncryptedIntegerType>();
  if (el1Ty == nullptr) {
    op->emitOpError() << "should have a !FHE.eint as the element type of the "
                         "tensor of operand #1";
    return mlir::failure();
  }
  if (el1Ty.getWidth() != el0Ty.getWidth()) {
    op->emitOpError() << "should have the width of encrypted equals"
                         ", got "
                      << el1Ty.getWidth() << " expect " << el0Ty.getWidth();
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult verifyTensorUnaryEint(mlir::Operation *op) {
  if (op->getNumOperands() != 1) {
    op->emitOpError() << "should have exactly 1 operands";
    return mlir::failure();
  }
  auto op0Ty = op->getOperand(0).getType().dyn_cast_or_null<mlir::TensorType>();
  if (op0Ty == nullptr) {
    op->emitOpError() << "should have operand as tensor";
    return mlir::failure();
  }
  auto el0Ty =
      op0Ty.getElementType()
          .dyn_cast_or_null<mlir::concretelang::FHE::EncryptedIntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have a !FHE.eint as the element type of the "
                         "tensor operand";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace impl

} // namespace OpTrait
} // namespace mlir

namespace mlir {
namespace concretelang {
namespace FHELinalg {

mlir::LogicalResult ApplyLookupTableEintOp::verify() {
  auto tTy = this->t().getType().cast<mlir::RankedTensorType>();
  auto tEltTy = tTy.getElementType()
                    .cast<mlir::concretelang::FHE::EncryptedIntegerType>();
  auto lutTy = this->lut().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();
  auto resultTy = this->getResult().getType().cast<mlir::RankedTensorType>();

  // Check the shape of lut argument
  auto tEltwidth = tEltTy.getWidth();
  mlir::SmallVector<int64_t, 1> expectedShape{1 << tEltwidth};
  if (!lutTy.hasStaticShape(expectedShape) || !lutEltTy.isInteger(64)) {
    this->emitOpError()
        << "should have as operand #2 a tensor<2^pxi64>, where p is the width "
           "of the encrypted integer of the operand #1,"
        << "expect tensor <" << expectedShape[0] << "xi64>";
    return mlir::failure();
  }
  if (!resultTy.hasStaticShape(tTy.getShape())) {
    this->emitOpError()
        << " should have same shapes for operand #1 and the result";
  }
  return mlir::success();
}

mlir::LogicalResult ApplyMultiLookupTableEintOp::verify() {
  auto tTy = this->t().getType().cast<mlir::RankedTensorType>();
  auto tEltTy = tTy.getElementType()
                    .cast<mlir::concretelang::FHE::EncryptedIntegerType>();
  auto lutTy = this->luts().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();
  auto resultTy = this->getResult().getType().cast<mlir::RankedTensorType>();

  // Check the shape of luts argument
  auto lut_size = lutTy.getShape()[lutTy.getShape().size() - 1];
  auto expected_lut_size = 1 << tEltTy.getWidth();
  if (lut_size != expected_lut_size || !lutEltTy.isInteger(64)) {
    this->emitOpError() << "should have as operand #2 a "
                           "tensor<DMx...xD1X2^pxi64>, where p is the width "
                           "of the encrypted integer of the operand #1,"
                        << "expect tensor <DMx...xD1X" << expected_lut_size
                        << "xi64>";
    return mlir::failure();
  }
  if (!resultTy.hasStaticShape(tTy.getShape())) {
    this->emitOpError()
        << " should have same shapes for operand #1 and the result";
  }
  return mlir::success();
}

mlir::RankedTensorType getTensorType(::mlir::Value value) {
  return value.getType().cast<mlir::RankedTensorType>();
}

template <class T> T getElmentType(::mlir::Value value) {
  auto tTy = getTensorType(value);
  return tTy.getElementType().cast<T>();
}

mlir::IntegerType getClearElmentType(::mlir::Value value) {
  return getElmentType<mlir::IntegerType>(value);
}

FHE::EncryptedIntegerType getEncryptedElmentType(::mlir::Value value) {
  using namespace mlir::concretelang::FHE;
  return getElmentType<FHE::EncryptedIntegerType>(value);
}

mlir::LogicalResult verifyMapHasRightShape(ApplyMappedLookupTableEintOp &op,
                                           ::mlir::Value &lut_input,
                                           ::mlir::Value &lut_map) {
  auto input_shape = getTensorType(lut_input).getShape();
  auto map_shape = getTensorType(lut_map).getShape();
  if (input_shape.equals(map_shape)) {
    return mlir::success();
  }
  std::string error;
  int input_rank = input_shape.size();
  int map_rank = map_shape.size();
  std::string input_name = "'t' (operand #1)";
  std::string map_name = "'lut_map.getName()' (operand #3)";
  if (input_rank == map_rank) {
    error = ": " + input_name + " dimensions differs from " + map_name;
  } else {
    error = ": " + input_name + " rank (=" + std::to_string(input_rank) +
            ") differs from " + map_name +
            " rank (=" + std::to_string(map_rank) + ")";
  }
  op.emitOpError() << error;
  return mlir::failure();
}

mlir::LogicalResult verifyLutsSize(ApplyMappedLookupTableEintOp &op,
                                   ::mlir::Value &encryptedIndex,
                                   ::mlir::Value &luts) {
  auto index_width = getEncryptedElmentType(encryptedIndex).getWidth();
  auto actual_lut_size = getTensorType(luts).getShape().back();
  auto expected_lut_size = 1 << index_width;
  if (actual_lut_size == expected_lut_size) {
    return mlir::success();
  }
  FHE::emitErrorBadLutSize(op, "luts", "ct", expected_lut_size, index_width);
  return mlir::failure();
}

mlir::LogicalResult ApplyMappedLookupTableEintOp::verify() {
  auto t = this->t();
  auto luts = this->luts();
  auto map = this->map();
  auto result = this->getResult();

  auto t_shape = getTensorType(t).getShape();
  if (!getTensorType(result).hasStaticShape(t_shape)) {
    this->emitOpError()
        << ": `t` (operand #1) and `map` (operand #2) must have the same shape";
    return mlir::failure();
  }

  if (!getTensorType(map).getElementType().isIndex()) {
    this->emitOpError()
        << ": `map` (operand #3) should contains elements of type `index`";
    return mlir::failure();
  }

  return mlir::success(verifyMapHasRightShape(*this, t, map).succeeded() &&
                       verifyLutsSize(*this, t, luts).succeeded());
}

::mlir::LogicalResult Dot::verify() {
  if (::mlir::failed(mlir::verifyCompatibleShape(this->lhs().getType(),
                                                 this->rhs().getType()))) {
    return this->emitOpError("arguments have incompatible shapes");
  }
  auto lhsEltType = this->lhs()
                        .getType()
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<FHE::EncryptedIntegerType>();
  auto rhsEltType = this->rhs()
                        .getType()
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<mlir::IntegerType>();
  auto resultType =
      this->getResult().getType().cast<FHE::EncryptedIntegerType>();
  if (!mlir::concretelang::FHE::
          verifyEncryptedIntegerAndIntegerInputsConsistency(
              *this->getOperation(), lhsEltType, rhsEltType)) {
    return ::mlir::failure();
  }
  if (!FHE::verifyEncryptedIntegerInputAndResultConsistency(
          *this->getOperation(), lhsEltType, resultType)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

llvm::SmallVector<int64_t, 3>
verifySumCalculateActualOutputShape(mlir::Type outputType) {
  auto actualOutputShape = llvm::SmallVector<int64_t, 3>{};
  if (outputType.isa<mlir::TensorType>()) {
    auto outputTensorType = outputType.dyn_cast<mlir::TensorType>();
    for (int64_t size : outputTensorType.getShape()) {
      actualOutputShape.push_back(size);
    }
  }
  return actualOutputShape;
}

llvm::SmallVector<int64_t, 3> verifySumCalculateExpectedOutputShape(
    llvm::ArrayRef<int64_t> inputShape, int64_t inputDimensions,
    std::unordered_set<int64_t> &axesToDestroy, bool keepDims) {

  auto expectedOutputShape = llvm::SmallVector<int64_t, 3>{};
  for (int64_t i = 0; i < inputDimensions; i++) {
    bool ithAxisIsDestroyed = axesToDestroy.find(i) != axesToDestroy.end();
    if (!ithAxisIsDestroyed) {
      expectedOutputShape.push_back(inputShape[i]);
    } else if (keepDims) {
      expectedOutputShape.push_back(1);
    }
  }
  return expectedOutputShape;
}

mlir::LogicalResult SumOp::verify() {
  mlir::Value input = this->getOperand();
  mlir::Value output = this->getResult();

  auto inputType = input.getType().dyn_cast<mlir::TensorType>();
  mlir::Type outputType = output.getType();

  FHE::EncryptedIntegerType inputElementType =
      inputType.getElementType().dyn_cast<FHE::EncryptedIntegerType>();
  FHE::EncryptedIntegerType outputElementType =
      !outputType.isa<mlir::TensorType>()
          ? outputType.dyn_cast<FHE::EncryptedIntegerType>()
          : outputType.dyn_cast<mlir::TensorType>()
                .getElementType()
                .dyn_cast<FHE::EncryptedIntegerType>();

  if (!FHE::verifyEncryptedIntegerInputAndResultConsistency(
          *this->getOperation(), inputElementType, outputElementType)) {
    return mlir::failure();
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputDimensions = (int64_t)inputShape.size();

  mlir::ArrayAttr axes = this->axes();
  bool keepDims = this->keep_dims();

  auto axesToDestroy = std::unordered_set<int64_t>{};
  for (mlir::Attribute axisAttribute : axes) {
    int64_t axis = axisAttribute.cast<mlir::IntegerAttr>().getInt();

    bool axisIsValid = (0 <= axis) && (axis < inputDimensions);
    if (!axisIsValid) {
      this->emitOpError("has invalid axes attribute");
      return mlir::failure();
    }

    axesToDestroy.insert(axis);
  }
  if (axesToDestroy.empty()) {
    for (int64_t i = 0; i < inputDimensions; i++) {
      axesToDestroy.insert(i);
    }
  }

  auto expectedOutputShape = verifySumCalculateExpectedOutputShape(
      inputShape, inputDimensions, axesToDestroy, keepDims);
  auto actualOutputShape = verifySumCalculateActualOutputShape(outputType);

  if (expectedOutputShape != actualOutputShape) {
    auto stream = this->emitOpError();

    stream << "does not have the proper output shape of <";
    if (!expectedOutputShape.empty()) {
      stream << expectedOutputShape[0];
      for (size_t i = 1; i < expectedOutputShape.size(); i++) {
        stream << "x" << expectedOutputShape[i];
      }
    }
    stream << ">";

    return mlir::failure();
  }

  return mlir::success();
}

static bool sameShapeExceptAxis(llvm::ArrayRef<int64_t> shape1,
                                llvm::ArrayRef<int64_t> shape2, size_t axis) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (i != axis && shape1[i] != shape2[i]) {
      return false;
    }
  }
  return true;
}

mlir::LogicalResult ConcatOp::verify() {
  unsigned numOperands = this->getNumOperands();
  if (numOperands < 2) {
    this->emitOpError() << "should have at least 2 inputs";
    return mlir::failure();
  }

  int64_t axis = this->axis();
  mlir::Value out = this->out();

  auto outVectorType = out.getType().dyn_cast<mlir::TensorType>();
  auto outElementType =
      outVectorType.getElementType().dyn_cast<FHE::EncryptedIntegerType>();

  llvm::ArrayRef<int64_t> outShape = outVectorType.getShape();
  size_t outDims = outShape.size();

  if (axis < 0 || (size_t)axis >= outDims) {
    this->emitOpError() << "has invalid axis attribute";
    return mlir::failure();
  }

  int64_t expectedOutputElementsInAxis = 0;

  size_t index = 0;
  for (mlir::Value in : this->ins()) {
    auto inVectorType = in.getType().dyn_cast<mlir::TensorType>();
    auto inElementType =
        inVectorType.getElementType().dyn_cast<FHE::EncryptedIntegerType>();
    if (!FHE::verifyEncryptedIntegerInputAndResultConsistency(
            *this->getOperation(), inElementType, outElementType)) {
      return ::mlir::failure();
    }

    llvm::ArrayRef<int64_t> inShape = inVectorType.getShape();
    if (!sameShapeExceptAxis(inShape, outShape, (size_t)axis)) {
      auto stream = this->emitOpError();

      stream << "does not have the proper shape of <";
      if (axis == 0) {
        stream << "?";
      } else {
        stream << outShape[0];
      }
      for (size_t i = 1; i < outDims; i++) {
        stream << "x";
        if (i == (size_t)axis) {
          stream << "?";
        } else {
          stream << outShape[i];
        }
      }
      stream << "> for input #" << index;

      return mlir::failure();
    }
    expectedOutputElementsInAxis += inShape[axis];

    index += 1;
  }

  if (outShape[axis] != expectedOutputElementsInAxis) {
    auto stream = this->emitOpError();

    stream << "does not have the proper output shape of <";
    if (axis == 0) {
      stream << expectedOutputElementsInAxis;
    } else {
      stream << outShape[0];
    }
    for (size_t i = 1; i < outDims; i++) {
      stream << "x";
      if (i == (size_t)axis) {
        stream << expectedOutputElementsInAxis;
      } else {
        stream << outShape[i];
      }
    }
    stream << ">";

    return mlir::failure();
  }

  return mlir::success();
}

/// Verify the matmul shapes, the type of tensor elements should be checked by
/// something else
template <typename MatMulOp> mlir::LogicalResult verifyMatmul(MatMulOp &op) {
  auto lhsType =
      ((mlir::Type)op.lhs().getType()).cast<mlir::RankedTensorType>();
  auto rhsType =
      ((mlir::Type)op.rhs().getType()).cast<mlir::RankedTensorType>();

  llvm::ArrayRef<int64_t> lhsShape = lhsType.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsType.getShape();

  int64_t lhsDims = (int64_t)lhsShape.size();
  int64_t rhsDims = (int64_t)rhsShape.size();

  auto expectedOutputShape = mlir::SmallVector<int64_t, 2>{};
  if (lhsDims == 2 && rhsDims == 2) {

    // MxN @ NxP -> MxP

    if (lhsShape[1] != rhsShape[0]) {
      op.emitOpError() << "should have the same size "
                          "on dimension #1 of operand #0 "
                          "and dimension #0 of operand #1";
      return mlir::failure();
    }

    expectedOutputShape.push_back(lhsShape[0]);
    expectedOutputShape.push_back(rhsShape[1]);

  } else if (lhsDims >= 2 && rhsDims >= 2) {

    // KxLxMxN @   NxP -> KxLxMxP
    // KxLxMxN @ LxNxP -> KxLxMxP
    // Kx1xMxN @ LxNxP -> KxLxMxP

    //   MxN @ KxLxNxP -> KxLxMxP
    // LxMxN @ KxLxNxP -> KxLxMxP
    // 1xMxN @ KxLxNxP -> KxLxMxP

    if (lhsShape[lhsDims - 1] != rhsShape[rhsDims - 2]) {
      op.emitOpError() << "should have the same size "
                       << "on dimension #" << lhsDims - 1 << " of operand #0 "
                       << "and dimension #" << rhsDims - 2 << " of operand #1";
      return mlir::failure();
    }

    auto expectedOutputShapeReversed = mlir::SmallVector<int64_t, 4>{};

    expectedOutputShapeReversed.push_back(rhsShape[rhsDims - 1]);
    expectedOutputShapeReversed.push_back(lhsShape[lhsDims - 2]);

    int64_t i = lhsDims - 3;
    int64_t j = rhsDims - 3;
    while (i >= 0 && j >= 0) {
      int64_t lhsSize = lhsShape[i];
      int64_t rhsSize = rhsShape[j];

      if (lhsSize == rhsSize || lhsSize == 1 || rhsSize == 1) {
        expectedOutputShapeReversed.push_back(std::max(lhsSize, rhsSize));
      } else {
        op.emitOpError() << "should have the same size or size of 1 "
                         << "on dimension #" << i << " of operand #0 "
                         << "and dimension #" << j << " of operand #1";
        return mlir::failure();
      }

      i--;
      j--;
    }
    while (i >= 0) {
      int64_t lhsSize = lhsShape[i];
      expectedOutputShapeReversed.push_back(lhsSize);
      i--;
    }
    while (j >= 0) {
      int64_t rhsSize = rhsShape[j];
      expectedOutputShapeReversed.push_back(rhsSize);
      j--;
    }

    while (!expectedOutputShapeReversed.empty()) {
      expectedOutputShape.push_back(expectedOutputShapeReversed.back());
      expectedOutputShapeReversed.pop_back();
    }

  } else if (lhsDims == 1 && rhsDims >= 2) {

    // N @     NxP ->     P
    // N @   LxNxP ->   LxP
    // N @ KxLxNxP -> KxLxP

    if (rhsShape[rhsDims - 2] != lhsShape[0]) {
      op.emitOpError() << "should have the same size "
                       << "on dimension #0 of operand #0 "
                       << "and dimension #" << rhsDims - 2 << " of operand #1";
      return mlir::failure();
    }

    for (int64_t i = 0; i < rhsDims; i++) {
      if (i != rhsDims - 2) {
        expectedOutputShape.push_back(rhsShape[i]);
      }
    }

  } else if (lhsDims >= 2 && rhsDims == 1) {

    //     MxN @ N ->     M
    //   LxMxN @ N ->   LxM
    // KxLxMxN @ N -> KxLxM

    if (lhsShape[lhsDims - 1] != rhsShape[0]) {
      op.emitOpError() << "should have the same size "
                       << "on dimension #" << lhsDims - 1 << " of operand #0 "
                       << "and dimension #0 of operand #1";
      return mlir::failure();
    }

    for (int64_t i = 0; i < lhsDims - 1; i++) {
      expectedOutputShape.push_back(lhsShape[i]);
    }

  } else {

    // M @ N

    op.emitOpError() << "should have at least one "
                        "multi dimensional tensor "
                        "as an operand";
    return mlir::failure();
  }

  auto resultType =
      ((mlir::Type)op.getResult().getType()).cast<mlir::RankedTensorType>();

  if (!resultType.hasStaticShape(expectedOutputShape)) {
    auto stream = op->emitOpError();

    stream << "does not have the proper output shape of ";

    stream << "<" << expectedOutputShape[0];
    for (size_t i = 1; i < expectedOutputShape.size(); i++) {
      stream << "x" << expectedOutputShape[i];
    }
    stream << ">";

    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult MatMulEintIntOp::verify() {
  return ::mlir::concretelang::FHELinalg::verifyMatmul<
      mlir::concretelang::FHELinalg::MatMulEintIntOp>(*this);
}

mlir::LogicalResult MatMulIntEintOp::verify() {
  return ::mlir::concretelang::FHELinalg::verifyMatmul<
      mlir::concretelang::FHELinalg::MatMulIntEintOp>(*this);
}

mlir::SmallVector<int64_t, 4>
getPaddingFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp) {
  mlir::SmallVector<int64_t, 4> paddingInts;
  llvm::Optional<mlir::DenseIntElementsAttr> optionalPadding = convOp.padding();
  if (optionalPadding.hasValue()) {
    auto paddingAttr = optionalPadding.getValue();
    auto paddingAttrShape =
        paddingAttr.getType().cast<RankedTensorType>().getShape();
    assert(paddingAttrShape.size() == 1 && paddingAttrShape[0] == 4 &&
           "incorrect padding shape");
    paddingInts.insert(paddingInts.begin(), paddingAttr.value_begin<int64_t>(),
                       paddingAttr.value_end<int64_t>());
  } else {
    paddingInts.insert(paddingInts.begin(), {0, 0, 0, 0});
  }
  return paddingInts;
}

mlir::SmallVector<int64_t, 2>
getStridesFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp) {
  mlir::SmallVector<int64_t, 2> stridesInts;
  llvm::Optional<mlir::DenseIntElementsAttr> optionalStrides = convOp.strides();
  if (optionalStrides.hasValue()) {
    auto stridesAttr = optionalStrides.getValue();
    auto stridesAttrShape =
        stridesAttr.getType().cast<RankedTensorType>().getShape();
    assert(stridesAttrShape.size() == 1 && stridesAttrShape[0] == 2 &&
           "incorrect strides shape");
    stridesInts.insert(stridesInts.begin(), stridesAttr.value_begin<int64_t>(),
                       stridesAttr.value_end<int64_t>());
  } else {
    stridesInts.insert(stridesInts.begin(), {1, 1});
  }
  return stridesInts;
}

mlir::SmallVector<int64_t, 2>
getDilationsFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp) {
  mlir::SmallVector<int64_t, 2> dilationsInts;
  llvm::Optional<mlir::DenseIntElementsAttr> optionalDilations =
      convOp.dilations();
  if (optionalDilations.hasValue()) {
    auto dilationsAttr = optionalDilations.getValue();
    auto dilationsAttrShape =
        dilationsAttr.getType().cast<RankedTensorType>().getShape();
    assert(dilationsAttrShape.size() == 1 && dilationsAttrShape[0] == 2 &&
           "incorrect dilations shape");
    dilationsInts.insert(dilationsInts.begin(),
                         dilationsAttr.value_begin<int64_t>(),
                         dilationsAttr.value_end<int64_t>());
  } else {
    dilationsInts.insert(dilationsInts.begin(), {1, 1});
  }
  return dilationsInts;
}

/// Verify the Conv2d shapes, attributes, and expected output dimensions
mlir::LogicalResult Conv2dOp::verify() {
  auto inputTy =
      ((mlir::Type)this->input().getType()).cast<mlir::RankedTensorType>();
  auto weightTy =
      ((mlir::Type)this->weight().getType()).cast<mlir::RankedTensorType>();
  auto resultTy =
      ((mlir::Type)this->getResult().getType()).cast<mlir::RankedTensorType>();
  auto inputShape = inputTy.getShape();
  auto weightShape = weightTy.getShape();
  auto resultShape = resultTy.getShape();

  auto p = inputTy.getElementType()
               .cast<mlir::concretelang::FHE::EncryptedIntegerType>()
               .getWidth();
  auto weightElementTyWidth =
      weightTy.getElementType().cast<mlir::IntegerType>().getWidth();
  if (weightElementTyWidth != p + 1) {
    this->emitOpError() << "expected weight element type to have width "
                        << p + 1 << " but got " << weightElementTyWidth;
    return mlir::failure();
  }

  // Checking dimensions
  if (inputShape.size() != 4) {
    this->emitOpError() << "input should have 4 dimensions (N*C*H*W) but got "
                        << inputShape.size();
    return mlir::failure();
  }
  if (weightShape.size() != 4) {
    this->emitOpError() << "weight should have 4 dimensions (F*C*H*W) but got "
                        << weightShape.size();
    return mlir::failure();
  }
  if (resultShape.size() != 4) {
    this->emitOpError() << "result should have 4 dimensions (N*C*H*W) but got "
                        << resultShape.size();
    return mlir::failure();
  }

  // Checking attributes
  mlir::SmallVector<int64_t, 4> paddingInts = getPaddingFromConv2d(*this);
  llvm::Optional<mlir::DenseIntElementsAttr> optionalPadding = this->padding();
  if (optionalPadding.hasValue()) {
    auto paddingAttr = optionalPadding.getValue();
    auto paddingAttrShape =
        paddingAttr.getType().cast<RankedTensorType>().getShape();
    if (paddingAttrShape.size() != 1 || paddingAttrShape[0] != 4) {
      this->emitOpError()
          << "padding should have a single dimension of size 4, but got shape ["
          << paddingAttrShape << "]";
      return mlir::failure();
    }
    for (auto i = 0; i < 4; i++) {
      // TODO: Support padding (#427)
      if (paddingInts[i] != 0) {
        this->emitOpError()
            << "padding isn't yet supported, but got a non zero value ("
            << paddingInts[i] << ") at index " << i;
        return mlir::failure();
      }

      if (paddingInts[i] < 0) {
        this->emitOpError() << "padding can't have a negative value, but got "
                            << paddingInts[i] << " at index " << i;
        return mlir::failure();
      }
    }
  }
  mlir::SmallVector<int64_t, 2> stridesInts = getStridesFromConv2d(*this);
  llvm::Optional<mlir::DenseIntElementsAttr> optionalStrides = this->strides();
  if (optionalStrides.hasValue()) {
    auto stridesAttr = optionalStrides.getValue();
    auto stridesAttrShape =
        stridesAttr.getType().cast<RankedTensorType>().getShape();
    if (stridesAttrShape.size() != 1 || stridesAttrShape[0] != 2) {
      this->emitOpError()
          << "strides should have a single dimension of size 2, but got shape ["
          << stridesAttrShape << "]";
      return mlir::failure();
    }
    for (auto i = 0; i < 2; i++) {
      if (stridesInts[i] < 1) {
        this->emitOpError()
            << "strides can't have a value less than 1, but got "
            << stridesInts[i] << " at index " << i;
        return mlir::failure();
      }
    }
  }
  mlir::SmallVector<int64_t, 2> dilationsInts = getDilationsFromConv2d(*this);
  llvm::Optional<mlir::DenseIntElementsAttr> optionalDilations =
      this->dilations();
  if (optionalDilations.hasValue()) {
    auto dilationsAttr = optionalDilations.getValue();
    auto dilationsAttrShape =
        dilationsAttr.getType().cast<RankedTensorType>().getShape();
    if (dilationsAttrShape.size() != 1 || dilationsAttrShape[0] != 2) {
      this->emitOpError() << "dilations should have a single dimension of "
                             "size 2, but got shape ["
                          << dilationsAttrShape << "]";
      return mlir::failure();
    }
    for (auto i = 0; i < 2; i++) {
      if (dilationsInts[i] < 1) {
        this->emitOpError()
            << "dilations can't have a value less than 1, but got "
            << dilationsInts[i] << " at index " << i;
        return mlir::failure();
      }
    }
  }

  // Extracting dimensions
  int64_t inputN = inputShape[0], inputC = inputShape[1],
          inputH = inputShape[2], inputW = inputShape[3];
  int64_t weightF = weightShape[0], weightC = weightShape[1],
          weightH = weightShape[2], weightW = weightShape[3];
  int64_t resultN = resultShape[0], resultC = resultShape[1],
          resultH = resultShape[2], resultW = resultShape[3];

  // Bias check if specified
  mlir::Value bias = this->bias();
  if (bias) {
    auto biasTy = ((mlir::Type)bias.getType()).cast<mlir::RankedTensorType>();
    auto biasShape = biasTy.getShape();
    if (biasShape.size() != 1) {
      this->emitOpError() << "bias should have 1 dimension but got "
                          << biasShape.size();
      return mlir::failure();
    }
    if (biasShape[0] != weightF) {
      this->emitOpError() << "expected bias vector to have size " << weightF
                          << " but got " << biasShape[0];
      return mlir::failure();
    }
    auto biasElementTyWidth =
        biasTy.getElementType().cast<mlir::IntegerType>().getWidth();
    if (biasElementTyWidth != p + 1) {
      this->emitOpError() << "expected bias element type to have width "
                          << p + 1 << " but got " << biasElementTyWidth;
      return mlir::failure();
    }
  }

  // Dimension sizes checks
  if (resultN != inputN) {
    this->emitOpError()
        << "expected result batch size to be equal to input batch size ("
        << inputN << ") but got " << resultN;
    return mlir::failure();
  }
  if (inputC != weightC) {
    this->emitOpError() << "expected number of channels in weight to be equal "
                           "to number of channels in input ("
                        << inputC << ") but got " << weightC;
    return mlir::failure();
  }
  if (weightF != resultC) {
    this->emitOpError() << "expected number of output channels to be equal to "
                           "the number of filters ("
                        << weightF << ") but got " << resultC;
    return mlir::failure();
  }

  int64_t paddingH = paddingInts[0] + paddingInts[2];
  int64_t paddingW = paddingInts[1] + paddingInts[3];
  int64_t dilationH = dilationsInts[0];
  int64_t dilationW = dilationsInts[1];
  int64_t strideH = stridesInts[0];
  int64_t strideW = stridesInts[1];
  int64_t expectedResultH =
      floor((inputH + paddingH - dilationH * (weightH - 1) - 1) / strideH) + 1;
  int64_t expectedResultW =
      floor((inputW + paddingW - dilationW * (weightW - 1) - 1) / strideW) + 1;

  if (expectedResultH != resultH) {
    this->emitOpError() << "expected height of output to be equal to "
                        << expectedResultH << " but got " << resultH;
    return mlir::failure();
  }
  if (expectedResultW != resultW) {
    this->emitOpError() << "expected width of output to be equal to "
                        << expectedResultW << " but got " << resultW;
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult FromElementOp::verify() {
  mlir::Value in = this->getOperand();
  mlir::Value out = this->getResult();

  auto inType = in.getType();
  auto outType = out.getType().dyn_cast<mlir::TensorType>();

  auto expectedOutType = outType.cloneWith({1}, inType);
  if (outType != expectedOutType) {
    this->emitOpError() << "has invalid output type (expected "
                        << expectedOutType << ", got " << outType << ")";
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Implementation of FhelinalgConv2DNchwFchwOp
// This is a generated functions from `make generate_conv_op`, and some helpers
// from LinalgOps.cpp
//===----------------------------------------------------------------------===//
using namespace mlir;
using namespace mlir::linalg;

ArrayAttr FhelinalgConv2DNchwFchwOp::iterator_types() {
  return Builder(getContext())
      .getStrArrayAttr(SmallVector<StringRef>{
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getReductionIteratorTypeName(), getReductionIteratorTypeName(),
          getReductionIteratorTypeName()});
}

static SmallVector<AffineExpr>
getSymbolBindings(FhelinalgConv2DNchwFchwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

  int64_t cst3 = self.strides().getValues<int64_t>()[{0}];
  exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

  int64_t cst5 = self.dilations().getValues<int64_t>()[{0}];
  exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

  int64_t cst7 = self.strides().getValues<int64_t>()[{1}];
  exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

  int64_t cst9 = self.dilations().getValues<int64_t>()[{1}];
  exprs.push_back(getAffineConstantExpr(cst9, context));

  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr FhelinalgConv2DNchwFchwOp::indexing_maps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(
      mlir::parseAttribute(
          "affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, "
          "s7, s8, s9, s10] -> (d0, d4, d2 * s3 + d5 * s5, d3 * s7 + d6 * s9)>",
          context)
          .cast<AffineMapAttr>()
          .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute(
                     "affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, "
                     "s4, s5, s6, s7, s8, s9, s10] -> (d1, d4, d5, d6)>",
                     context)
                     .cast<AffineMapAttr>()
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute(
                     "affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, "
                     "s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>",
                     context)
                     .cast<AffineMapAttr>()
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned FhelinalgConv2DNchwFchwOp::getNumRegionArgs() { return 3; }

std::string FhelinalgConv2DNchwFchwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool FhelinalgConv2DNchwFchwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult FhelinalgConv2DNchwFchwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();

  if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
    if (!attr.getType().getElementType().isInteger(64))
      return op->emitError("incorrect element type for indexing map required "
                           "attribute 'strides'");
    if (attr.getType().getShape() != ArrayRef<int64_t>{2})
      return op->emitError(
          "incorrect shape for indexing map required attribute 'strides'");
  } else {
    return op->emitError("missing indexing map required attribute 'strides'");
  }

  if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
    if (!attr.getType().getElementType().isInteger(64))
      return op->emitError("incorrect element type for indexing map required "
                           "attribute 'dilations'");
    if (attr.getType().getShape() != ArrayRef<int64_t>{2})
      return op->emitError(
          "incorrect shape for indexing map required attribute 'dilations'");
  } else {
    return op->emitError("missing indexing map required attribute 'dilations'");
  }

  return success();
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty())
    return;
  p.printOptionalArrowTypeList(resultTypes);
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static void printNamedStructuredOp(OpAsmPrinter &p, Operation *op,
                                   ValueRange inputs, ValueRange outputs) {
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operand_segment_sizes",
                       // See generated code in mlir-linalg-yaml-gen.cpp
                       "linalg.memoized_indexing_maps"});

  // Printing is shared with generic ops, except for the region and
  // attributes.
  printCommonStructuredOpParts(p, inputs, outputs);

  // Results printing.
  printNamedStructuredOpResults(p, op->getResultTypes());

  // Region is elided.
}

void FhelinalgConv2DNchwFchwOp::print(mlir::OpAsmPrinter &p) {
  printNamedStructuredOp(p, this->getOperation(),
                         this->getOperation()->getOperands(),
                         this->getOperation()->getResults());
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  parser.parseOptionalAttrDict(result.attributes);

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(inputsOperands.size()),
                           static_cast<int32_t>(outputsOperands.size())}));
  return success();
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static ParseResult parseNamedStructuredOpRegion(
    OpAsmParser &parser, Region &region, unsigned numRegionArgs,
    TypeRange inputTypes, TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    RegionBuilderFn regionBuilder) {
  if (numRegionArgs != inputTypes.size() + outputTypes.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                      "region expects {0} args, got {1}",
                      numRegionArgs, inputTypes.size() + outputTypes.size()));
  }

  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, region, inputTypes, outputTypes, attrs,
                         regionBuilder);
  return success();
}

// Copied from LinalgOps.cpp; license is: Apache License v2.0 with
// LLVM Exceptions
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result,
                                          unsigned numRegionArgs,
                                          RegionBuilderFn regionBuilder) {
  // TODO: Enable when ods-gen supports captures.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // TODO: consider merging results parsing into region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, numRegionArgs, inputTypes,
                                   outputTypes, result.attributes.getAttrs(),
                                   regionBuilder))
    return failure();
  result.addRegion(std::move(region));

  return success();
}

mlir::ParseResult
FhelinalgConv2DNchwFchwOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  return parseNamedStructuredOp(parser, result, getNumRegionArgs(),
                                getRegionBuilder());
}

/// Some helpers were copied from LinalgOps.cpp

/// Generic entry point to create the block for the region of a LinalgOp.
/// This is used by both named structured ops created by ods-gen and by manually
/// defined C++ ops.
/// This is used by both builders and parsers.
/// This function creates the block in the region with arguments corresponding
/// to the elemental types of `inputTypes` and `outputTypes`. The latter are
/// asserted to be of ShapedType.
template <typename NamedStructuredOpType>
static void fillStructuredOpRegion(
    OpBuilder &opBuilder, Region &region, TypeRange inputTypes,
    TypeRange outputTypes,
    std::function<void(unsigned, unsigned)> errorHandler = nullptr);

/// Generic entry point to create both the region and the block of a LinalgOp.
template <typename NamedStructuredOpType>
static void
createAndFillStructuredOpRegion(OpBuilder &opBuilder, OperationState &result,
                                TypeRange inputTypes, TypeRange outputTypes);

/// Common parsing and printing used for both named structured ops created by
/// ods-gen and by manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes);
template <typename NamedStructuredOpType>
static void printCommonStructuredOpParts(OpAsmPrinter &p,
                                         NamedStructuredOpType op);

/// Specific parsing and printing for named structured ops created by ods-gen.
template <typename NamedStructuredOpType>
static ParseResult
parseNamedStructuredOpRegion(OpAsmParser &parser, Region &region,
                             TypeRange inputTypes, TypeRange outputTypes);

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes);

template <typename NamedStructuredOpType>
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result);

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes);

template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op);

class RegionBuilderHelper {
public:
  RegionBuilderHelper(MLIRContext *context, Block &block)
      : context(context), block(block) {}

  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();

    if (operand.getType() == toType)
      return operand;
    if (auto toIntType = toType.dyn_cast<IntegerType>()) {
      // If operand is floating point, cast directly to the int type.
      if (operand.getType().isa<FloatType>()) {
        if (isUnsignedCast)
          return builder.create<arith::FPToUIOp>(loc, toType, operand);
        return builder.create<arith::FPToSIOp>(loc, toType, operand);
      }
      // Cast index operands directly to the int type.
      if (operand.getType().isIndex())
        return builder.create<arith::IndexCastOp>(loc, toType, operand);
      if (auto fromIntType = operand.getType().dyn_cast<IntegerType>()) {
        // Either extend or truncate.
        if (toIntType.getWidth() > fromIntType.getWidth()) {
          if (isUnsignedCast)
            return builder.create<arith::ExtUIOp>(loc, toType, operand);
          return builder.create<arith::ExtSIOp>(loc, toType, operand);
        }
        if (toIntType.getWidth() < fromIntType.getWidth())
          return builder.create<arith::TruncIOp>(loc, toType, operand);
      }
    } else if (auto toFloatType = toType.dyn_cast<FloatType>()) {
      // If operand is integer, cast directly to the float type.
      // Note that it is unclear how to cast from BF16<->FP16.
      if (operand.getType().isa<IntegerType>()) {
        if (isUnsignedCast)
          return builder.create<arith::UIToFPOp>(loc, toFloatType, operand);
        return builder.create<arith::SIToFPOp>(loc, toFloatType, operand);
      }
      if (auto fromFloatType = operand.getType().dyn_cast<FloatType>()) {
        if (toFloatType.getWidth() > fromFloatType.getWidth())
          return builder.create<arith::ExtFOp>(loc, toFloatType, operand);
        if (toFloatType.getWidth() < fromFloatType.getWidth())
          return builder.create<arith::TruncFOp>(loc, toFloatType, operand);
      }
    }

    emitWarning(operand.getLoc()) << "could not cast operand of type "
                                  << operand.getType() << " to " << toType;
    return operand;
  }

  Value applyfn__add(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::AddFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs) &&
        rhs.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
      return builder.create<mlir::concretelang::FHE::AddEintIntOp>(lhs.getLoc(),
                                                                   rhs, lhs);
    }
    if (lhs.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>() &&
        isInteger(rhs)) {
      return builder.create<mlir::concretelang::FHE::AddEintIntOp>(lhs.getLoc(),
                                                                   lhs, rhs);
    }
    if (lhs.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>() &&
        rhs.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
      return builder.create<mlir::concretelang::FHE::AddEintOp>(lhs.getLoc(),
                                                                lhs, rhs);
    }
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__exp(Value x) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(x))
      return builder.create<math::ExpOp>(x.getLoc(), x);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__log(Value x) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(x))
      return builder.create<math::LogOp>(x.getLoc(), x);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__sub(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::SubFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::SubIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__mul(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MulFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MulIOp>(lhs.getLoc(), lhs, rhs);
    if (lhs.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>() &&
        isInteger(rhs)) {
      return builder.create<mlir::concretelang::FHE::MulEintIntOp>(lhs.getLoc(),
                                                                   lhs, rhs);
    }
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__max(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MaxFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MaxSIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__max_unsigned(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MaxFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MaxUIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__min(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MinFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MinSIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  Value applyfn__min_unsigned(Value lhs, Value rhs) {
    OpBuilder builder = getBuilder();
    if (isFloatingPoint(lhs))
      return builder.create<arith::MinFOp>(lhs.getLoc(), lhs, rhs);
    if (isInteger(lhs))
      return builder.create<arith::MinUIOp>(lhs.getLoc(), lhs, rhs);
    llvm_unreachable("unsupported non numeric type");
  }

  void yieldOutputs(ValueRange values) {
    assert(!values.empty() && "linalg ops must yield outputs");
    if (values.empty())
      return;
    Value first = values.front();
    OpBuilder builder = getBuilder();
    builder.create<YieldOp>(first.getLoc(), values);
  }

  Value constant(std::string value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return builder.create<arith::ConstantOp>(loc, valueAttr.getType(),
                                             valueAttr);
  }

  Value index(int64_t dim) {
    OpBuilder builder = getBuilder();
    return builder.create<IndexOp>(builder.getUnknownLoc(), dim);
  }

  Type getIntegerType(unsigned width) {
    return IntegerType::get(context, width);
  }

  Type getFloat32Type() { return Float32Type::get(context); }

  Type getFloat64Type() { return Float64Type::get(context); }

private:
  MLIRContext *context;
  Block &block;

  bool isFloatingPoint(Value value) { return value.getType().isa<FloatType>(); }
  bool isInteger(Value value) { return value.getType().isa<IntegerType>(); }

  OpBuilder getBuilder() {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(&block);
    return builder;
  }
};

static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

/// Generic entry point to create the block for the region of a LinalgOp.
/// This is used by both named structured ops created by ods-gen and by manually
/// defined C++ ops.
/// This is used by both builders and parsers.
/// This function creates the block in the region with arguments corresponding
/// to the elemental types of `inputTypes` and `outputTypes`, which are asserted
/// to be ShapedType.
template <typename NamedStructuredOpType>
static void
fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                       TypeRange inputTypes, TypeRange outputTypes,
                       std::function<void(unsigned, unsigned)> errorHandler) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes})
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));
      argLocs.push_back(opBuilder.getUnknownLoc());
    }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  unsigned actual = body->getNumArguments();
  unsigned expected = NamedStructuredOpType::getNumRegionArgs();
  if (expected != actual) {
    if (errorHandler)
      errorHandler(expected, actual);
    return;
  }

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  NamedStructuredOpType::regionBuilder(b, *body, {});

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputs) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputs) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

/// Generic entry point to create both the region and the block of a LinalgOp.
template <typename NamedStructuredOpType>
void createAndFillStructuredOpRegion(OpBuilder &opBuilder,
                                     OperationState &result,
                                     TypeRange inputTypes,
                                     TypeRange outputTypes) {
  Region &region = *result.addRegion();
  fillStructuredOpRegion<NamedStructuredOpType>(
      opBuilder, region, inputTypes, outputTypes,
      [&](unsigned expected, unsigned actual) {
        assert(expected != actual && "incorrect number of arguments");
      });
}

/// END OF COPY FROM LinalgOps.cpp

void FhelinalgConv2DNchwFchwOp::regionBuilder(
    ImplicitLocOpBuilder &b, Block &block,
    llvm::ArrayRef<mlir::NamedAttribute>) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "FhelinalgConv2DNchwFchwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value value1 =
      helper.cast(block.getArgument(0).getType(), block.getArgument(0), false);
  Value value2 =
      helper.cast(block.getArgument(1).getType(), block.getArgument(1), false);
  Value value3 = helper.applyfn__mul(value1, value2);
  Value value4 = helper.applyfn__add(block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

LogicalResult FhelinalgConv2DNchwFchwOp::fold(ArrayRef<Attribute>,
                                              SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
void FhelinalgConv2DNchwFchwOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getGenericEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                        outputBuffers);
}

/// Verify the transpose shapes
mlir::LogicalResult TransposeOp::verify() {
  mlir::Type tensorTy = ((mlir::Type)this->tensor().getType());
  if (!tensorTy.isa<RankedTensorType>()) {
    this->emitOpError() << "should have operand as tensor";
    return mlir::failure();
  }
  mlir::Type resultTy = ((mlir::Type)this->getResult().getType());
  if (!resultTy.isa<RankedTensorType>()) {
    this->emitOpError() << "should have result as tensor";
    return mlir::failure();
  }
  auto tensorShapedTy = tensorTy.dyn_cast_or_null<mlir::ShapedType>();
  auto resultShapedTy = resultTy.dyn_cast_or_null<mlir::ShapedType>();
  if (tensorShapedTy.getShape().size() != resultShapedTy.getShape().size()) {
    this->emitOpError()
        << "input and output tensors should have the same number of dimensions";
    return mlir::failure();
  }
  if (tensorShapedTy.getElementType() != resultShapedTy.getElementType()) {
    this->emitOpError()
        << "input and output tensors should have the same element type";
    return mlir::failure();
  }
  size_t n_dims = tensorShapedTy.getShape().size();
  for (size_t i = 0; i < n_dims; i++) {
    if (tensorShapedTy.getDimSize(i) !=
        resultShapedTy.getDimSize(n_dims - (i + 1))) {
      this->emitOpError()
          << "output tensor should have inverted dimensions of input";
      return mlir::failure();
    }
  }
  return mlir::success();
}

/// Avoid addition with constant tensor of 0s
OpFoldResult AddEintIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2);
  auto toAdd = operands[1].dyn_cast_or_null<mlir::DenseIntElementsAttr>();
  if (toAdd == nullptr)
    return nullptr;
  for (auto it = toAdd.begin(); it != toAdd.end(); it++) {
    if (*it != 0) {
      return nullptr;
    }
  }
  return getOperand(0);
}

/// Avoid subtraction with constant tensor of 0s
OpFoldResult SubEintIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2);
  auto toSub = operands[1].dyn_cast_or_null<mlir::DenseIntElementsAttr>();
  if (toSub == nullptr)
    return nullptr;
  for (auto it = toSub.begin(); it != toSub.end(); it++) {
    if (*it != 0) {
      return nullptr;
    }
  }
  return getOperand(0);
}

/// Avoid multiplication with constant tensor of 1s
OpFoldResult MulEintIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2);
  auto toMul = operands[1].dyn_cast_or_null<mlir::DenseIntElementsAttr>();
  if (toMul == nullptr)
    return nullptr;
  for (auto it = toMul.begin(); it != toMul.end(); it++) {
    if (*it != 1) {
      return nullptr;
    }
  }
  return getOperand(0);
}

} // namespace FHELinalg
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.cpp.inc"
