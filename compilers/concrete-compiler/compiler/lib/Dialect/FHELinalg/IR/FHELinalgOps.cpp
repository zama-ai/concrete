// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <unordered_set>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.h"
#include "concretelang/Support/CompilerEngine.h"
#include "llvm/ADT/SmallVector.h"

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
          .dyn_cast_or_null<mlir::concretelang::FHE::FheIntegerInterface>();
  if (el0Ty == nullptr) {
    op->emitOpError()
        << "should have !FHE.eint or !FHE.esint as the element type of the "
           "tensor of operand #0";
    return mlir::failure();
  }

  auto el1Ty = op1Ty.getElementType().dyn_cast_or_null<mlir::IntegerType>();
  if (el1Ty == nullptr) {
    op->emitOpError() << "should have an integer as the element type of the "
                         "tensor of operand #1";
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
          .dyn_cast_or_null<mlir::concretelang::FHE::FheIntegerInterface>();
  if (el1Ty == nullptr) {
    op->emitOpError()
        << "should have !FHE.eint or !FHE.esint as the element type of the "
           "tensor of operand #1";
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
          .dyn_cast_or_null<mlir::concretelang::FHE::FheIntegerInterface>();
  if (el0Ty == nullptr) {
    op->emitOpError()
        << "should have !FHE.eint or !FHE.esint as the element type of the "
           "tensor of operand #0";
    return mlir::failure();
  }

  auto el1Ty =
      op1Ty.getElementType()
          .dyn_cast_or_null<mlir::concretelang::FHE::FheIntegerInterface>();
  if (el1Ty == nullptr) {
    op->emitOpError()
        << "should have !FHE.eint or !FHE.esint as the element type of the "
           "tensor of operand #1";
    return mlir::failure();
  }

  if (el0Ty.isSigned() != el1Ty.isSigned()) {
    op->emitOpError()
        << "should have the signedness of encrypted arguments equal";
    return mlir::failure();
  }

  unsigned el0BitWidth = el0Ty.getWidth();
  unsigned el1BitWidth = el1Ty.getWidth();

  if (el1BitWidth != el0BitWidth) {
    op->emitOpError() << "should have the width of encrypted equals"
                         ", got "
                      << el1BitWidth << " expect " << el0BitWidth;
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
          .dyn_cast_or_null<mlir::concretelang::FHE::FheIntegerInterface>();
  if (el0Ty == nullptr) {
    op->emitOpError()
        << "should have !FHE.eint or !FHE.esint as the element type of the "
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
  auto tTy = this->getT().getType().cast<mlir::RankedTensorType>();
  auto tEltTy =
      tTy.getElementType().cast<mlir::concretelang::FHE::FheIntegerInterface>();
  auto lutTy = this->getLut().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();
  auto resultTy = this->getResult().getType().cast<mlir::RankedTensorType>();

  // Check the shape of lut argument
  auto tEltwidth = tEltTy.getWidth();
  mlir::SmallVector<int64_t, 1> expectedShape{1 << tEltwidth};
  if (!lutTy.hasStaticShape(expectedShape) || !lutEltTy.isSignlessInteger() ||
      lutEltTy.getIntOrFloatBitWidth() > 64) {
    this->emitOpError() << "should have as operand #2 a "
                           "tensor<2^pxi{8,16,32,64}>, where p is the width "
                           "of the encrypted integer of the operand #1,"
                        << "expect tensor <" << expectedShape[0]
                        << "xi{8,16,32,64}>";
    return mlir::failure();
  }
  if (!resultTy.hasStaticShape(tTy.getShape())) {
    this->emitOpError()
        << " should have same shapes for operand #1 and the result";
  }
  return mlir::success();
}

mlir::LogicalResult ApplyMultiLookupTableEintOp::verify() {
  auto tTy = this->getT().getType().cast<mlir::RankedTensorType>();
  auto tEltTy =
      tTy.getElementType().cast<mlir::concretelang::FHE::FheIntegerInterface>();
  auto lutTy = this->getLuts().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();
  auto resultTy = this->getResult().getType().cast<mlir::RankedTensorType>();

  // Check the shape of luts argument
  auto lut_size = lutTy.getShape()[lutTy.getShape().size() - 1];
  auto expected_lut_size = 1 << tEltTy.getWidth();
  if (lut_size != expected_lut_size || !lutEltTy.isSignlessInteger() ||
      lutEltTy.getIntOrFloatBitWidth() > 64) {
    this->emitOpError()
        << "should have as operand #2 a "
           "tensor<DMx...xD1X2^pxi{8,16,32,64}>, where p is the width "
           "of the encrypted integer of the operand #1,"
        << "expect tensor <DMx...xD1X" << expected_lut_size
        << "xi{8,16,32,64}>";
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

FHE::FheIntegerInterface getEncryptedElmentType(::mlir::Value value) {
  using namespace mlir::concretelang::FHE;
  return getElmentType<FHE::FheIntegerInterface>(value);
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
  auto t = this->getT();
  auto tTy = this->getT().getType().cast<mlir::RankedTensorType>();
  auto tEltTy =
      tTy.getElementType().cast<mlir::concretelang::FHE::FheIntegerInterface>();
  auto luts = this->getLuts();
  auto map = this->getMap();
  auto result = this->getResult();
  auto lutTy = this->getLuts().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();

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

  auto expected_lut_size = 1 << tEltTy.getWidth();
  if (!lutEltTy.isSignlessInteger() || lutEltTy.getIntOrFloatBitWidth() > 64) {
    this->emitOpError()
        << "should have as operand #2 a "
           "tensor<DMx...xD1X2^pxi{8,16,32,64}>, where p is the width "
           "of the encrypted integer of the operand #1,"
        << "expect tensor <DMx...xD1X" << expected_lut_size
        << "xi{8,16,32,64}>";
    return mlir::failure();
  }

  return mlir::success(verifyMapHasRightShape(*this, t, map).succeeded() &&
                       verifyLutsSize(*this, t, luts).succeeded());
}

mlir::LogicalResult
verifyDotInputsOutputsConsistency(mlir::concretelang::FHELinalg::DotEint &op,
                                  FHE::FheIntegerInterface &lhsEltType,
                                  FHE::FheIntegerInterface &rhsEltType,
                                  FHE::FheIntegerInterface &resultType) {
  if (!mlir::concretelang::FHE::verifyEncryptedIntegerInputsConsistency(
          *op.getOperation(), lhsEltType, rhsEltType)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

mlir::LogicalResult
verifyDotInputsOutputsConsistency(mlir::concretelang::FHELinalg::Dot &op,
                                  FHE::FheIntegerInterface &lhsEltType,
                                  mlir::IntegerType &rhsEltType,
                                  FHE::FheIntegerInterface &resultType) {
  if (!FHE::verifyEncryptedIntegerInputAndResultConsistency(
          *op.getOperation(), lhsEltType, resultType)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

// Verify a dot product operation:
// - check that the shapes are compatible
// - check that the widths of the inputs and result is the same
template <typename DotOp, typename RHSElementType>
mlir::LogicalResult verifyDot(DotOp &op) {
  if (::mlir::failed(mlir::verifyCompatibleShape(op.getLhs().getType(),
                                                 op.getRhs().getType()))) {
    return op.emitOpError("arguments have incompatible shapes");
  }
  auto lhsEltType = ((mlir::Type)op.getLhs().getType())
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .dyn_cast<FHE::FheIntegerInterface>();
  auto rhsEltType = ((mlir::Type)op.getRhs().getType())
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<RHSElementType>();
  auto resultType = ((mlir::Type)op.getResult().getType())
                        .dyn_cast<FHE::FheIntegerInterface>();

  return verifyDotInputsOutputsConsistency(op, lhsEltType, rhsEltType,
                                           resultType);
}

::mlir::LogicalResult Dot::verify() {
  return ::mlir::concretelang::FHELinalg::verifyDot<
      mlir::concretelang::FHELinalg::Dot, mlir::IntegerType>(*this);
}

::mlir::LogicalResult DotEint::verify() {
  return ::mlir::concretelang::FHELinalg::verifyDot<
      mlir::concretelang::FHELinalg::DotEint, FHE::FheIntegerInterface>(*this);
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
  Type outputType = output.getType();

  auto inputElementType =
      inputType.getElementType().dyn_cast<FHE::FheIntegerInterface>();
  auto outputElementType = !outputType.isa<mlir::TensorType>()
                               ? outputType.dyn_cast<FHE::FheIntegerInterface>()
                               : outputType.dyn_cast<mlir::TensorType>()
                                     .getElementType()
                                     .dyn_cast<FHE::FheIntegerInterface>();

  if (!FHE::verifyEncryptedIntegerInputAndResultConsistency(
          *this->getOperation(), inputElementType, outputElementType)) {
    return mlir::failure();
  }

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputDimensions = (int64_t)inputShape.size();

  mlir::ArrayAttr axes = this->getAxes();
  bool keepDims = this->getKeepDims();

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

OpFoldResult ConcatOp::fold(FoldAdaptor operands) {
  if (this->getNumOperands() == 1) {
    return this->getOperand(0);
  }
  return nullptr;
}

mlir::LogicalResult ConcatOp::verify() {
  unsigned numOperands = this->getNumOperands();
  if (numOperands < 1) {
    this->emitOpError() << "should have at least 1 input";
    return mlir::failure();
  }

  int64_t axis = this->getAxis();
  mlir::Value out = this->getOut();

  auto outVectorType = out.getType().dyn_cast<mlir::TensorType>();
  auto outElementType = outVectorType.getElementType();

  llvm::ArrayRef<int64_t> outShape = outVectorType.getShape();
  size_t outDims = outShape.size();

  if (axis < 0 || (size_t)axis >= outDims) {
    this->emitOpError() << "has invalid axis attribute";
    return mlir::failure();
  }

  int64_t expectedOutputElementsInAxis = 0;

  size_t index = 0;
  for (mlir::Value in : this->getIns()) {
    auto inVectorType = in.getType().dyn_cast<mlir::TensorType>();
    auto inElementType = inVectorType.getElementType();
    if (inElementType != outElementType) {
      this->emitOpError() << "input element type (" << inElementType
                          << ") doesn't match output element type ("
                          << outElementType << ")";
      return mlir::failure();
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
      ((mlir::Type)op.getLhs().getType()).cast<mlir::RankedTensorType>();
  auto rhsType =
      ((mlir::Type)op.getRhs().getType()).cast<mlir::RankedTensorType>();

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

mlir::LogicalResult MatMulEintEintOp::verify() {
  return ::mlir::concretelang::FHELinalg::verifyMatmul<
      mlir::concretelang::FHELinalg::MatMulEintEintOp>(*this);
}

mlir::SmallVector<int64_t, 4>
getPaddingFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp) {
  mlir::SmallVector<int64_t, 4> paddingInts;
  std::optional<mlir::DenseIntElementsAttr> optionalPadding =
      convOp.getPadding();
  if (optionalPadding.has_value()) {
    auto paddingAttr = optionalPadding.value();
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
  std::optional<mlir::DenseIntElementsAttr> optionalStrides =
      convOp.getStrides();
  if (optionalStrides.has_value()) {
    auto stridesAttr = optionalStrides.value();
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
  std::optional<mlir::DenseIntElementsAttr> optionalDilations =
      convOp.getDilations();
  if (optionalDilations.has_value()) {
    auto dilationsAttr = optionalDilations.value();
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

int64_t getGroupFromConv2d(mlir::concretelang::FHELinalg::Conv2dOp &convOp) {
  std::optional<uint64_t> optionalGroup = convOp.getGroup();
  if (optionalGroup.has_value())
    return optionalGroup.value();
  return 1;
}

/// Verify the Conv2d shapes, attributes, and expected output dimensions
mlir::LogicalResult Conv2dOp::verify() {
  auto inputTy =
      ((mlir::Type)this->getInput().getType()).cast<mlir::RankedTensorType>();
  auto weightTy =
      ((mlir::Type)this->getWeight().getType()).cast<mlir::RankedTensorType>();
  auto resultTy =
      ((mlir::Type)this->getResult().getType()).cast<mlir::RankedTensorType>();
  auto inputShape = inputTy.getShape();
  auto weightShape = weightTy.getShape();
  auto resultShape = resultTy.getShape();

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
  std::optional<mlir::DenseIntElementsAttr> optionalPadding =
      this->getPadding();
  if (optionalPadding.has_value()) {
    auto paddingAttr = optionalPadding.value();
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
  std::optional<mlir::DenseIntElementsAttr> optionalStrides =
      this->getStrides();
  if (optionalStrides.has_value()) {
    auto stridesAttr = optionalStrides.value();
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
  std::optional<mlir::DenseIntElementsAttr> optionalDilations =
      this->getDilations();
  if (optionalDilations.has_value()) {
    auto dilationsAttr = optionalDilations.value();
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
  int64_t group = getGroupFromConv2d(*this);
  if (group < 1) {
    this->emitOpError() << "group must be strictly positif, but got " << group;
    return mlir::failure();
  }

  // Extracting dimensions
  int64_t inputN = inputShape[0], inputC = inputShape[1],
          inputH = inputShape[2], inputW = inputShape[3];
  int64_t weightF = weightShape[0], weightC = weightShape[1],
          weightH = weightShape[2], weightW = weightShape[3];
  int64_t resultN = resultShape[0], resultC = resultShape[1],
          resultH = resultShape[2], resultW = resultShape[3];

  // Bias check if specified
  mlir::Value bias = this->getBias();
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
  }

  // Dimension sizes checks
  if (resultN != inputN) {
    this->emitOpError()
        << "expected result batch size to be equal to input batch size ("
        << inputN << ") but got " << resultN;
    return mlir::failure();
  }
  if (weightC != inputC / group) {
    this->emitOpError()
        << "expected number of channels in weight to be equal to "
        << inputC / group << " (input_channels / group) but got " << weightC;
    return mlir::failure();
  }
  if (weightF % group != 0) {
    this->emitOpError() << "expected number of feature maps (" << weightF
                        << ") to be a multiple of group (" << group << ")";
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

mlir::LogicalResult Maxpool2dOp::verify() {
  const mlir::RankedTensorType inputTy =
      this->getInput().getType().cast<mlir::RankedTensorType>();
  const mlir::RankedTensorType outputTy =
      this->getResult().getType().cast<mlir::RankedTensorType>();

  const FHE::FheIntegerInterface inputElementTy = inputTy.getElementType();
  const FHE::FheIntegerInterface outputElementTy = outputTy.getElementType();

  if (inputElementTy != outputElementTy) {
    this->emitOpError() << "expected output element type "
                        << "(" << outputElementTy << ") "
                        << "to be the same with input element type "
                        << "(" << inputElementTy << ") "
                        << "but it is not";
    return mlir::failure();
  }

  const llvm::ArrayRef<int64_t> inputShape = inputTy.getShape();
  const llvm::ArrayRef<int64_t> outputShape = outputTy.getShape();

  if (inputShape.size() != 4) {
    this->emitOpError() << "expected input to have 4 dimensions (N*C*H*W) "
                        << "but it has " << inputShape.size();
    return mlir::failure();
  }
  if (outputShape.size() != 4) {
    this->emitOpError() << "expected output to have 4 dimensions (N*C*H*W) "
                        << "but it has " << outputShape.size();
    return mlir::failure();
  }

  const int64_t inputN = inputShape[0];
  const int64_t inputC = inputShape[1];
  const int64_t inputH = inputShape[2];
  const int64_t inputW = inputShape[3];

  const mlir::DenseIntElementsAttr kernelShapeAttr = this->getKernelShape();
  const mlir::RankedTensorType kernelShapeAttrTy =
      kernelShapeAttr.getType().cast<mlir::RankedTensorType>();
  const llvm::ArrayRef<int64_t> kernelShapeAttrShape =
      kernelShapeAttrTy.getShape();

  if (kernelShapeAttrShape.size() != 1 || kernelShapeAttrShape[0] != 2) {
    this->emitOpError() << "expected kernel shape to be of shape "
                        << "(2) "
                        << "but it is of shape "
                        << "(" << kernelShapeAttrShape << ")";
    return mlir::failure();
  }

  mlir::SmallVector<int64_t, 2> kernelShape;
  kernelShape.append(kernelShapeAttr.value_begin<int64_t>(),
                     kernelShapeAttr.value_end<int64_t>());

  const int64_t kernelShapeH = kernelShape[0];
  const int64_t kernelShapeW = kernelShape[1];

  mlir::SmallVector<int64_t, 2> strides;
  const llvm::Optional<mlir::DenseIntElementsAttr> maybeStridesAttr =
      this->getStrides();
  if (maybeStridesAttr.has_value()) {
    const mlir::DenseIntElementsAttr stridesAttr = maybeStridesAttr.value();
    const mlir::RankedTensorType stridesAttrTy =
        stridesAttr.getType().cast<mlir::RankedTensorType>();
    const llvm::ArrayRef<int64_t> stridesAttrShape = stridesAttrTy.getShape();

    if (stridesAttrShape.size() != 1 || stridesAttrShape[0] != 2) {
      this->emitOpError() << "expected strides to be of shape "
                          << "(2) "
                          << "but it is of shape "
                          << "(" << stridesAttrShape << ")";
      return mlir::failure();
    }

    strides.append(stridesAttr.value_begin<int64_t>(),
                   stridesAttr.value_end<int64_t>());
  } else {
    strides.append({1, 1});
  }
  for (size_t i = 0; i < 2; i++) {
    if (strides[i] < 1) {
      this->emitOpError() << "expected elements of strides to be positive "
                          << "but strides[" << i << "] is " << strides[i];
      return mlir::failure();
    }
  }

  const int64_t stridesH = strides[0];
  const int64_t stridesW = strides[1];

  mlir::SmallVector<int64_t, 2> dilations;
  const llvm::Optional<mlir::DenseIntElementsAttr> maybeDilationsAttr =
      this->getDilations();
  if (maybeDilationsAttr.has_value()) {
    const mlir::DenseIntElementsAttr dilationsAttr = maybeDilationsAttr.value();
    const mlir::RankedTensorType dilationsAttrTy =
        dilationsAttr.getType().cast<mlir::RankedTensorType>();
    const llvm::ArrayRef<int64_t> dilationsAttrShape =
        dilationsAttrTy.getShape();

    if (dilationsAttrShape.size() != 1 || dilationsAttrShape[0] != 2) {
      this->emitOpError() << "expected dilations to be of shape "
                          << "(2) "
                          << "but it is of shape "
                          << "(" << dilationsAttrShape << ")";
      return mlir::failure();
    }

    dilations.append(dilationsAttr.value_begin<int64_t>(),
                     dilationsAttr.value_end<int64_t>());
  } else {
    dilations.append({1, 1});
  }
  for (size_t i = 0; i < 2; i++) {
    if (dilations[i] < 1) {
      this->emitOpError() << "expected elements of dilations to be positive "
                          << "but dilations[" << i << "] is " << dilations[i];
      return mlir::failure();
    }
  }

  const int64_t dilationsH = dilations[0];
  const int64_t dilationsW = dilations[1];

  const int64_t expectedOutputH =
      floor((inputH - dilationsH * (kernelShapeH - 1) - 1) / stridesH) + 1;
  const int64_t expectedOutputW =
      floor((inputW - dilationsW * (kernelShapeW - 1) - 1) / stridesW) + 1;
  const mlir::SmallVector<int64_t, 4> expectedOutputShape = {
      inputN,
      inputC,
      expectedOutputH,
      expectedOutputW,
  };

  if (outputShape != llvm::ArrayRef(expectedOutputShape)) {
    this->emitOpError() << "expected output to be of shape "
                        << "(" << expectedOutputShape << ") "
                        << "but it is of shape "
                        << "(" << outputShape << ")";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult FromElementOp::verify() {
  mlir::Value in = this->getOperand();
  mlir::Value out = this->getResult();

  auto inType = in.getType();
  auto outType = out.getType().dyn_cast<mlir::TensorType>();

  llvm::SmallVector<int64_t> shape{1};
  auto expectedOutType = outType.cloneWith(std::optional{shape}, inType);

  if (outType != expectedOutType) {
    this->emitOpError() << "has invalid output type (expected "
                        << expectedOutType << ", got " << outType << ")";
    return mlir::failure();
  }

  return mlir::success();
}

/// Verify the transpose shapes
mlir::LogicalResult TransposeOp::verify() {
  mlir::Type tensorTy = ((mlir::Type)this->getTensor().getType());
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

  llvm::ArrayRef<int64_t> inShape = tensorShapedTy.getShape();
  llvm::ArrayRef<int64_t> outShape = resultShapedTy.getShape();

  int64_t inputDimensions = (int64_t)inShape.size();

  mlir::ArrayAttr axes = this->getAxes();
  if (axes.empty()) {
    for (int64_t i = 0; i < inputDimensions; i++) {
      if (inShape[i] != outShape[inputDimensions - (i + 1)]) {
        this->emitOpError()
            << "output tensor should have inverted dimensions of input";
        return mlir::failure();
      }
    }
  } else {
    if (axes.size() != (size_t)inputDimensions) {
      this->emitOpError() << "has invalid axes attribute (doesn't have "
                          << inputDimensions << " elements)";
      return mlir::failure();
    }

    auto seenAxes = std::unordered_set<int64_t>{};

    size_t i = 0;
    for (mlir::Attribute axisAttribute : axes) {
      int64_t axis = axisAttribute.cast<mlir::IntegerAttr>().getInt();

      bool axisIsValid = (0 <= axis) && (axis < inputDimensions);
      if (!axisIsValid) {
        this->emitOpError()
            << "has invalid axes attribute (axes[" << i << "] "
            << "isn't in range [0, " << inputDimensions - 1 << "])";
        return mlir::failure();
      }

      seenAxes.insert(axis);

      if (outShape[i] != inShape[axis]) {
        this->emitOpError() << "has invalid output shape (output.shape[" << i
                            << "] is not input.shape[axes[" << i << "]])";
        return mlir::failure();
      }

      i++;
    }
    if (seenAxes.size() != (size_t)inputDimensions) {
      this->emitOpError()
          << "has invalid axes attribute (doesn't contain all input axes)";
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult ToSignedOp::verify() {
  auto inputType = this->getInput().getType().cast<mlir::ShapedType>();
  auto outputType = this->getResult().getType().cast<mlir::ShapedType>();

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  if (inputShape != outputShape) {
    this->emitOpError()
        << "input and output tensors should have the same shape";
    return mlir::failure();
  }

  auto inputElementType =
      inputType.getElementType().cast<FHE::EncryptedUnsignedIntegerType>();
  auto outputElementType =
      outputType.getElementType().cast<FHE::EncryptedSignedIntegerType>();

  if (inputElementType.getWidth() != outputElementType.getWidth()) {
    this->emitOpError()
        << "input and output tensors should have the same width";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ToUnsignedOp::verify() {
  mlir::ShapedType inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::ShapedType>();
  mlir::ShapedType outputType =
      this->getResult().getType().dyn_cast_or_null<mlir::ShapedType>();

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  if (inputShape != outputShape) {
    this->emitOpError()
        << "input and output tensors should have the same shape";
    return mlir::failure();
  }

  auto inputElementType =
      inputType.getElementType().cast<FHE::EncryptedSignedIntegerType>();
  auto outputElementType =
      outputType.getElementType().cast<FHE::EncryptedUnsignedIntegerType>();

  if (inputElementType.getWidth() != outputElementType.getWidth()) {
    this->emitOpError()
        << "input and output tensors should have the same width";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult RoundOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  if (inputShape != outputShape) {
    this->emitOpError()
        << "input and output tensors should have the same shape";
    return mlir::failure();
  }

  auto inputElementType =
      inputType.getElementType().cast<FHE::FheIntegerInterface>();
  auto outputElementType =
      outputType.getElementType().cast<FHE::FheIntegerInterface>();

  if (inputElementType.getWidth() < outputElementType.getWidth()) {
    this->emitOpError()
        << "input tensor should have bigger bit width than output tensor";
    return mlir::failure();
  }

  if (inputElementType.isSigned() != outputElementType.isSigned()) {
    this->emitOpError()
        << "input and output tensors should have the same signedness";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult LsbEintOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  if (inputShape != outputShape) {
    this->emitOpError()
        << "input and output tensors should have the same shape";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ReinterpretPrecisionEintOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  if (inputShape != outputShape) {
    this->emitOpError()
        << "input and output tensors should have the same shape";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ChangePartitionEintOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  if (inputShape != outputShape) {
    this->emitOpError()
        << "input and output tensors should have the same shape";
    return mlir::failure();
  }

  auto inputElementType =
      inputType.getElementType().cast<FHE::FheIntegerInterface>();
  auto outputElementType =
      outputType.getElementType().cast<FHE::FheIntegerInterface>();
  if (!FHE::verifyEncryptedIntegerInputAndResultConsistency(
          *this->getOperation(), inputElementType, outputElementType)) {
    return mlir::failure();
  }

  if (!FHE::verifyPartitionConsistency(this)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult FancyIndexOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto indicesType =
      this->getIndices().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputElementType = inputType.getElementType();
  auto outputElementType = outputType.getElementType();

  if (inputElementType != outputElementType) {
    this->emitOpError() << "input element type " << inputElementType
                        << " doesn't match output element type "
                        << outputElementType;
    return mlir::failure();
  }

  auto inputShape = inputType.getShape();
  auto indicesShape = indicesType.getShape();
  auto outputShape = outputType.getShape();

  auto inputIsVector = inputShape.size() == 1;
  if (!inputIsVector) {
    if (indicesShape[indicesShape.size() - 1] != (int64_t)inputShape.size()) {
      this->emitOpError()
          << "size of the last dimension of indices '"
          << indicesShape[indicesShape.size() - 1]
          << "' doesn't match the number of dimensions of input '"
          << inputShape.size() << "'";
      return mlir::failure();
    }
  }

  auto expectedOutputShape =
      inputIsVector ? indicesShape : indicesShape.drop_back();
  if (outputShape != expectedOutputShape) {
    auto stream = this->emitOpError();

    stream << "output shape '<";
    if (!outputShape.empty()) {
      llvm::interleave(outputShape, stream, "x");
    }
    stream << ">' doesn't match the expected output shape '<";
    if (!expectedOutputShape.empty()) {
      llvm::interleave(expectedOutputShape, stream, "x");
    }
    stream << ">'";

    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult FancyAssignOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto indicesType =
      this->getIndices().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto valuesType =
      this->getValues().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputElementType = inputType.getElementType();
  auto valuesElementType = valuesType.getElementType();
  auto outputElementType = outputType.getElementType();

  if (valuesElementType != inputElementType) {
    this->emitOpError() << "values element type " << valuesElementType
                        << " doesn't match input element type "
                        << inputElementType;
    return mlir::failure();
  }
  if (outputElementType != inputElementType) {
    this->emitOpError() << "output element type " << outputElementType
                        << " doesn't match input element type "
                        << inputElementType;
    return mlir::failure();
  }

  auto inputShape = inputType.getShape();
  auto indicesShape = indicesType.getShape();
  auto valuesShape = valuesType.getShape();
  auto outputShape = outputType.getShape();

  auto inputIsVector = inputShape.size() == 1;
  if (!inputIsVector) {
    if (indicesShape[indicesShape.size() - 1] != (int64_t)inputShape.size()) {
      this->emitOpError()
          << "size of the last dimension of indices '"
          << indicesShape[indicesShape.size() - 1]
          << "' doesn't match the number of dimensions of input '"
          << inputShape.size() << "'";
      return mlir::failure();
    }
  }

  auto expectedValuesShape =
      inputIsVector ? indicesShape
                    : indicesShape.slice(0, indicesShape.size() - 1);
  if (valuesShape != expectedValuesShape) {
    auto stream = this->emitOpError();

    stream << "values shape '<";
    if (!valuesShape.empty()) {
      llvm::interleave(valuesShape, stream, "x");
    }
    stream << ">' doesn't match the expected values shape '<";
    if (!expectedValuesShape.empty()) {
      llvm::interleave(expectedValuesShape, stream, "x");
    }
    stream << ">'";

    return mlir::failure();
  }

  auto expectedOutputShape = inputShape;
  if (outputShape != expectedOutputShape) {
    auto stream = this->emitOpError();

    stream << "output shape '<";
    if (!outputShape.empty()) {
      llvm::interleave(outputShape, stream, "x");
    }
    stream << ">' doesn't match the expected output shape '<";
    if (!expectedOutputShape.empty()) {
      llvm::interleave(expectedOutputShape, stream, "x");
    }
    stream << ">'";

    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult BroadcastOp::verify() {
  auto inputType =
      this->getInput().getType().dyn_cast<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast<mlir::RankedTensorType>();

  auto inputElementType = inputType.getElementType();
  auto outputElementType = outputType.getElementType();

  if (inputElementType != outputElementType) {
    this->emitOpError() << "input element type " << inputElementType
                        << " doesn't match output element type "
                        << outputElementType;
    return mlir::failure();
  }

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  auto inputDimensions = inputShape.size();
  auto outputDimensions = outputShape.size();

  auto cannot_be_done = false;
  if (inputDimensions > outputDimensions) {
    cannot_be_done = true;
    return mlir::failure();
  } else {
    auto modifiedInputShape =
        llvm::SmallVector<int64_t>(outputDimensions - inputDimensions, 1);
    modifiedInputShape.append(inputShape.begin(), inputShape.end());

    for (size_t i = 0; i < outputDimensions; i++) {
      if (modifiedInputShape[i] != outputShape[i] &&
          modifiedInputShape[i] != 1) {
        cannot_be_done = true;
        break;
      }
    }
  }

  if (cannot_be_done) {
    auto stream = this->emitOpError();

    stream << "input shape '<";
    if (!inputShape.empty()) {
      llvm::interleave(inputShape, stream, "x");
    }
    stream << ">' cannot be broadcasted to output shape '<";
    if (!outputShape.empty()) {
      llvm::interleave(outputShape, stream, "x");
    }
    stream << ">'";

    return mlir::failure();
  }

  return mlir::success();
}

/// Avoid addition with constant tensor of 0s
OpFoldResult AddEintIntOp::fold(FoldAdaptor operands) {
  auto toAdd = operands.getRhs().dyn_cast_or_null<mlir::DenseIntElementsAttr>();
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
OpFoldResult SubEintIntOp::fold(FoldAdaptor operands) {
  auto toSub = operands.getRhs().dyn_cast_or_null<mlir::DenseIntElementsAttr>();
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
OpFoldResult MulEintIntOp::fold(FoldAdaptor operands) {
  auto toMul = operands.getRhs().dyn_cast_or_null<mlir::DenseIntElementsAttr>();
  if (toMul == nullptr)
    return nullptr;
  for (auto it = toMul.begin(); it != toMul.end(); it++) {
    if (*it != 1) {
      return nullptr;
    }
  }
  return getOperand(0);
}

void MulEintIntOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {

  // Replace multiplication by clear zero cst to a trivial encrypted zero tensor
  class ZeroCstOpPattern : public mlir::OpRewritePattern<MulEintIntOp> {
  public:
    ZeroCstOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<MulEintIntOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(MulEintIntOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp = op.getRhs().getDefiningOp<arith::ConstantOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      auto vals = cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value");
      for (auto it = vals.begin(); it != vals.end(); it++) {
        if (*it != 0) {
          return mlir::failure();
        }
      }
      rewriter.replaceOpWithNewOp<FHE::ZeroTensorOp>(op,
                                                     op.getResult().getType());
      return mlir::success();
    }
  };

  // Replace multiplication by encrypted zero cst to a trivial encrypted zero
  // tensor
  class ZeroEncOpPattern : public mlir::OpRewritePattern<MulEintIntOp> {
  public:
    ZeroEncOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<MulEintIntOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(MulEintIntOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp = op.getLhs().getDefiningOp<FHE::ZeroTensorOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      rewriter.replaceAllUsesWith(op, cstOp);
      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
  patterns.add<ZeroCstOpPattern>(context);
  patterns.add<ZeroEncOpPattern>(context);
}

/// Avoid multiplication with constant tensor of 1s
OpFoldResult RoundOp::fold(FoldAdaptor operands) {

  auto input = this->getInput();
  auto inputType =
      this->getInput().getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto outputType =
      this->getOutput().getType().dyn_cast_or_null<mlir::RankedTensorType>();

  auto inputElementType =
      inputType.getElementType().cast<FHE::FheIntegerInterface>();
  auto outputElementType =
      outputType.getElementType().cast<FHE::FheIntegerInterface>();

  if (inputElementType.getWidth() == outputElementType.getWidth()) {
    return input;
  }
  return nullptr;
}

template <typename MatMulOp>
void getMatMulCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::MLIRContext *context) {

  // Replace multiplication by clear zero cst to a trivial encrypted zero tensor
  class ZeroCstOpPattern : public mlir::OpRewritePattern<MatMulOp> {
  public:
    ZeroCstOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<MatMulOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(MatMulOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp =
          op.getClearMatrix().template getDefiningOp<arith::ConstantOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      auto vals =
          cstOp->template getAttrOfType<mlir::DenseIntElementsAttr>("value");
      for (auto it = vals.begin(); it != vals.end(); it++) {
        if (*it != 0) {
          return mlir::failure();
        }
      }
      rewriter.replaceOpWithNewOp<FHE::ZeroTensorOp>(op,
                                                     op.getResult().getType());
      return mlir::success();
    }
  };

  // Replace multiplication by encrypted zero cst to a trivial encrypted zero
  // tensor
  class ZeroEncOpPattern : public mlir::OpRewritePattern<MatMulOp> {
  public:
    ZeroEncOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<MatMulOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(MatMulOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp =
          op.getEncryptedMatrix().template getDefiningOp<FHE::ZeroTensorOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<FHE::ZeroTensorOp>(op,
                                                     op.getResult().getType());
      return mlir::success();
    }
  };
  patterns.add<ZeroCstOpPattern>(context);
  patterns.add<ZeroEncOpPattern>(context);
}

void MatMulIntEintOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  getMatMulCanonicalizationPatterns<MatMulIntEintOp>(patterns, context);
}

void MatMulEintIntOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  getMatMulCanonicalizationPatterns<MatMulEintIntOp>(patterns, context);
}

template <typename SignedConvOp>
void getSignedConvCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *context) {
  // Replace to_signed of zero to signed zero
  class ZeroOpPattern : public mlir::OpRewritePattern<SignedConvOp> {
  public:
    ZeroOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<SignedConvOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(SignedConvOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp = op.getInput().template getDefiningOp<FHE::ZeroTensorOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<FHE::ZeroTensorOp>(op,
                                                     op.getResult().getType());
      return mlir::success();
    }
  };
  patterns.add<ZeroOpPattern>(context);
}

void ToSignedOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                             mlir::MLIRContext *context) {
  getSignedConvCanonicalizationPatterns<ToSignedOp>(patterns, context);
}

void ToUnsignedOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  getSignedConvCanonicalizationPatterns<ToUnsignedOp>(patterns, context);
}

std::optional<mlir::Value>
fuseBackToBackTableLookups(mlir::Operation *currentOperation,
                           mlir::PatternRewriter &rewriter) {

  using mlir::concretelang::FHE::FheIntegerInterface;

  CompilationOptions currentCompilationOptions = getCurrentCompilationOptions();

  if (!currentCompilationOptions.enableTluFusing) {
    return std::nullopt;
  }

  auto currentOperationAsTlu =
      llvm::dyn_cast<ApplyLookupTableEintOp>(currentOperation);
  auto currentOperationAsMappedTlu =
      llvm::dyn_cast<ApplyMappedLookupTableEintOp>(currentOperation);

  if (!currentOperationAsTlu && !currentOperationAsMappedTlu) {
    return std::nullopt;
  }

  auto intermediateValue =
      (currentOperationAsTlu ? (currentOperationAsTlu.getT())
                             : (currentOperationAsMappedTlu.getT()));

  auto intermediateOperation = intermediateValue.getDefiningOp();
  if (!intermediateOperation) {
    return std::nullopt;
  }

  auto intermediateOperationAsTlu =
      llvm::dyn_cast<ApplyLookupTableEintOp>(intermediateOperation);
  auto intermediateOperationAsMappedTlu =
      llvm::dyn_cast<ApplyMappedLookupTableEintOp>(intermediateOperation);

  if (!intermediateOperationAsTlu && !intermediateOperationAsMappedTlu) {
    return std::nullopt;
  }

  auto inputValue =
      (intermediateOperationAsTlu ? (intermediateOperationAsTlu.getT())
                                  : (intermediateOperationAsMappedTlu.getT()));

  struct Indexer {
    int64_t tableSize;
    bool isSigned;

    Indexer(int64_t tableSize, bool isSigned)
        : tableSize{tableSize}, isSigned{isSigned} {}

    virtual ~Indexer() = default;

    int64_t sanitizeIndex(int64_t index) const {
      // Same logic as the lookup lambda in
      // cannonicalization of FHE.apply_lookup_table.
      // See FHEOps.cpp for explanation of the following code.
      if (index < 0) {
        index += tableSize;
        if (index < 0) {
          index = tableSize / 2;
        }
      } else if (index >= tableSize) {
        if (!isSigned) {
          index = tableSize - 1;
        } else {
          index = (tableSize / 2) - 1;
        }
      }
      return index;
    }

    virtual int64_t get(int64_t index, int64_t position) const = 0;
  };

  struct TluIdexer : public Indexer {
    std::vector<int64_t> tableContent;

    TluIdexer(int64_t tableSize, bool isSigned,
              std::vector<int64_t> tableContent)
        :

          Indexer{tableSize, isSigned}, tableContent{std::move(tableContent)} {}

    ~TluIdexer() override = default;

    static std::optional<std::unique_ptr<TluIdexer>>
    create(ApplyLookupTableEintOp operation) {
      auto tableValue = operation.getLut();

      auto tableOperation =
          llvm::dyn_cast_or_null<arith::ConstantOp>(tableValue.getDefiningOp());
      if (!tableOperation) {
        return std::nullopt;
      }

      auto tableContentAttr =
          tableOperation.getValueAttr()
              .dyn_cast_or_null<mlir::DenseIntElementsAttr>();
      if (!tableContentAttr) {
        return std::nullopt;
      }

      auto tableContent = std::vector<int64_t>();
      for (auto value : tableContentAttr.getValues<int64_t>()) {
        tableContent.push_back(value);
      }

      auto inputValue = operation.getT();
      auto inputType = inputValue.getType()
                           .cast<RankedTensorType>()
                           .getElementType()
                           .dyn_cast<FheIntegerInterface>();

      auto tableSize = 1 << inputType.getWidth();
      auto isSigned = inputType.isSigned();

      return std::make_unique<TluIdexer>(
          TluIdexer(tableSize, isSigned, tableContent));
    };

    int64_t get(int64_t index, int64_t position) const override {
      return tableContent[sanitizeIndex(index)];
    }
  };

  struct MappedTluIdexer : public Indexer {
    std::vector<int64_t> tablesContent;
    std::vector<int64_t> mapContent;

    MappedTluIdexer(int64_t tableSize, bool isSigned,
                    std::vector<int64_t> tablesContent,
                    std::vector<int64_t> mapContent)
        :

          Indexer{tableSize, isSigned}, tablesContent{std::move(tablesContent)},
          mapContent{std::move(mapContent)} {}

    ~MappedTluIdexer() override = default;

    static std::optional<std::unique_ptr<MappedTluIdexer>>
    create(ApplyMappedLookupTableEintOp operation) {
      auto tablesValue = operation.getLuts();

      auto tablesOperation = llvm::dyn_cast_or_null<arith::ConstantOp>(
          tablesValue.getDefiningOp());
      if (!tablesOperation) {
        return std::nullopt;
      }

      auto tablesContentAttr =
          tablesOperation.getValueAttr()
              .dyn_cast_or_null<mlir::DenseIntElementsAttr>();
      if (!tablesContentAttr) {
        return std::nullopt;
      }

      auto tablesContent = std::vector<int64_t>();
      for (auto value : tablesContentAttr.getValues<int64_t>()) {
        tablesContent.push_back(value);
      }

      auto mapValue = operation.getMap();

      auto mapOperation =
          llvm::dyn_cast_or_null<arith::ConstantOp>(mapValue.getDefiningOp());
      if (!mapOperation) {
        return std::nullopt;
      }

      auto mapContentAttr = mapOperation.getValueAttr()
                                .dyn_cast_or_null<mlir::DenseIntElementsAttr>();
      if (!mapContentAttr) {
        return std::nullopt;
      }

      auto mapContent = std::vector<int64_t>();
      for (auto value : mapContentAttr.getValues<int64_t>()) {
        mapContent.push_back(value);
      }

      auto inputValue = operation.getT();
      auto inputType = inputValue.getType()
                           .cast<RankedTensorType>()
                           .getElementType()
                           .dyn_cast<FheIntegerInterface>();

      auto tableSize = 1 << inputType.getWidth();
      auto isSigned = inputType.isSigned();

      return std::make_unique<MappedTluIdexer>(
          MappedTluIdexer(tableSize, isSigned, tablesContent, mapContent));
    }

    int64_t get(int64_t index, int64_t position) const override {
      int64_t tableIndex = mapContent[position];
      return tablesContent[sanitizeIndex(index) + (tableIndex * tableSize)];
    }
  };

  std::unique_ptr<Indexer> intermediateIndexer;
  if (intermediateOperationAsTlu) {
    auto indexer = TluIdexer::create(intermediateOperationAsTlu);
    if (!indexer) {
      return std::nullopt;
    }
    intermediateIndexer = std::move(*indexer);
  } else {
    auto indexer = MappedTluIdexer::create(intermediateOperationAsMappedTlu);
    if (!indexer) {
      return std::nullopt;
    }
    intermediateIndexer = std::move(*indexer);
  }

  std::unique_ptr<Indexer> currentIndexer;
  if (currentOperationAsTlu) {
    auto indexer = TluIdexer::create(currentOperationAsTlu);
    if (!indexer) {
      return std::nullopt;
    }
    currentIndexer = std::move(*indexer);
  } else {
    auto indexer = MappedTluIdexer::create(currentOperationAsMappedTlu);
    if (!indexer) {
      return std::nullopt;
    }
    currentIndexer = std::move(*indexer);
  }

  auto usersOfPreviousOperation = intermediateOperation->getUsers();
  auto numberOfUsersOfPreviousOperation = std::distance(
      usersOfPreviousOperation.begin(), usersOfPreviousOperation.end());

  if (numberOfUsersOfPreviousOperation > 1) {
    // This is a special case.
    //
    // Imagine you have this structure:
    // -----------------
    // x: uint6
    // y: uint3 = tlu[x]
    // z: uint3 = y + 1
    // a: uint3 = tlu[y]
    // b: uint3 = a + z
    // -----------------
    //
    // In this case, it might be better not to fuse `a = tlu[tlu[x]]`.
    //
    // The reason is, intermediate `y` is necessary for `z`,
    // so it have to be computed anyway.
    //
    // So to calculate `a`, there are 2 options:
    // - fused tlu on x
    // - regular tlu on y
    //
    // In this case, it's best to fuse only if
    // bit width of `x` is smaller than bit width of `y`.

    // We can use the table size as it's derived from the bit width
    // and it preserves the ordering.
    auto xTableSize = intermediateIndexer->tableSize;
    auto yTableSize = currentIndexer->tableSize;

    auto shouldFuse = xTableSize < yTableSize;
    if (!shouldFuse) {
      return std::nullopt;
    }
  }

  auto resultingType =
      (currentOperationAsTlu ? (currentOperationAsTlu.getType())
                             : (currentOperationAsMappedTlu.getType()));

  if (intermediateOperationAsTlu && currentOperationAsTlu) {
    auto newTableContent = std::vector<int64_t>();
    newTableContent.reserve(intermediateIndexer->tableSize);

    if (!intermediateIndexer->isSigned) {
      for (ssize_t x = 0; x < intermediateIndexer->tableSize; x++) {
        auto resultOfFirstTableLookup = intermediateIndexer->get(x, 0);
        newTableContent.push_back(
            currentIndexer->get(resultOfFirstTableLookup, 0));
      }
    } else {
      for (ssize_t x = 0; x < intermediateIndexer->tableSize / 2; x++) {
        auto resultOfFirstTableLookup = intermediateIndexer->get(x, 0);
        newTableContent.push_back(
            currentIndexer->get(resultOfFirstTableLookup, 0));
      }
      for (ssize_t x = -(intermediateIndexer->tableSize / 2); x < 0; x++) {
        auto resultOfFirstTableLookup = intermediateIndexer->get(x, 0);
        newTableContent.push_back(
            currentIndexer->get(resultOfFirstTableLookup, 0));
      }
    }

    auto newTableShape = std::vector<int64_t>{intermediateIndexer->tableSize};
    auto newTableType = RankedTensorType::get(
        newTableShape, IntegerType::get(currentOperation->getContext(), 64));

    auto newTable = rewriter.create<arith::ConstantOp>(
        currentOperation->getLoc(),
        DenseIntElementsAttr::get(newTableType, newTableContent));

    auto newOperation = rewriter.create<ApplyLookupTableEintOp>(
        currentOperation->getLoc(), resultingType, inputValue, newTable);

    return newOperation;
  }

  auto newTableContents = std::vector<std::vector<int64_t>>();
  auto newMapContent = std::vector<int64_t>();

  auto inputShape = inputValue.getType().cast<RankedTensorType>().getShape();
  int64_t numberOfInputs = 1;
  for (auto dimension : inputShape) {
    numberOfInputs *= dimension;
  }

  for (int64_t position = 0; position < numberOfInputs; position++) {
    auto newTableContent = std::vector<int64_t>();
    newTableContent.reserve(intermediateIndexer->tableSize);

    if (!intermediateIndexer->isSigned) {
      for (ssize_t x = 0; x < intermediateIndexer->tableSize; x++) {
        auto resultOfFirstTableLookup = intermediateIndexer->get(x, position);
        newTableContent.push_back(
            currentIndexer->get(resultOfFirstTableLookup, position));
      }
    } else {
      for (ssize_t x = 0; x < intermediateIndexer->tableSize / 2; x++) {
        auto resultOfFirstTableLookup = intermediateIndexer->get(x, position);
        newTableContent.push_back(
            currentIndexer->get(resultOfFirstTableLookup, position));
      }
      for (ssize_t x = -(intermediateIndexer->tableSize / 2); x < 0; x++) {
        auto resultOfFirstTableLookup = intermediateIndexer->get(x, position);
        newTableContent.push_back(
            currentIndexer->get(resultOfFirstTableLookup, position));
      }
    }

    auto search = std::find(newTableContents.begin(), newTableContents.end(),
                            newTableContent);

    size_t index;
    if (search == newTableContents.end()) {
      index = newTableContents.size();
      newTableContents.push_back(newTableContent);
    } else {
      index = std::distance(newTableContents.begin(), search);
    }

    newMapContent.push_back(index);
  }

  if (newTableContents.size() == 1) {
    auto newTableShape = std::vector<int64_t>{intermediateIndexer->tableSize};
    auto newTableType = RankedTensorType::get(
        newTableShape, IntegerType::get(currentOperation->getContext(), 64));

    auto newTable = rewriter.create<arith::ConstantOp>(
        currentOperation->getLoc(),
        DenseIntElementsAttr::get(newTableType, newTableContents[0]));

    auto newOperation = rewriter.create<ApplyLookupTableEintOp>(
        currentOperation->getLoc(), resultingType, inputValue, newTable);

    return newOperation;
  } else {
    auto newTablesShape =
        std::vector<int64_t>{static_cast<int64_t>(newTableContents.size()),
                             intermediateIndexer->tableSize};
    auto newTablesType = RankedTensorType::get(
        newTablesShape, IntegerType::get(currentOperation->getContext(), 64));

    auto newTableContentsFlattened = std::vector<int64_t>();
    for (auto newTableContent : newTableContents) {
      newTableContentsFlattened.insert(newTableContentsFlattened.end(),
                                       newTableContent.begin(),
                                       newTableContent.end());
    }

    auto newTables = rewriter.create<arith::ConstantOp>(
        currentOperation->getLoc(),
        DenseIntElementsAttr::get(newTablesType, newTableContentsFlattened));

    auto newMapShape = inputShape;
    auto newMapType = RankedTensorType::get(
        newMapShape, IndexType::get(currentOperation->getContext()));

    auto newMap = rewriter.create<arith::ConstantOp>(
        currentOperation->getLoc(),
        DenseIntElementsAttr::get(newMapType, newMapContent));

    auto newOperation = rewriter.create<ApplyMappedLookupTableEintOp>(
        currentOperation->getLoc(), resultingType, inputValue, newTables,
        newMap);

    return newOperation;
  }

  return std::nullopt;
}

void ApplyLookupTableEintOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {

  class AfterTluPattern
      : public mlir::OpRewritePattern<ApplyLookupTableEintOp> {
  public:
    AfterTluPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<ApplyLookupTableEintOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(ApplyLookupTableEintOp currentOperation,
                    mlir::PatternRewriter &rewriter) const override {
      auto replacement = fuseBackToBackTableLookups(currentOperation, rewriter);
      if (replacement) {
        CompilationOptions currentCompilationOptions =
            getCurrentCompilationOptions();
        if (currentCompilationOptions.printTluFusing) {
          printTluFusing(currentOperation.getT(),
                         currentOperation->getResult(0), *replacement);
        }

        rewriter.replaceAllUsesWith(currentOperation, *replacement);
        return mlir::success();
      }
      return mlir::failure();
    }
  };
  patterns.add<AfterTluPattern>(context);
}

void ApplyMappedLookupTableEintOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {

  class AfterTluPattern
      : public mlir::OpRewritePattern<ApplyMappedLookupTableEintOp> {
  public:
    AfterTluPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<ApplyMappedLookupTableEintOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(ApplyMappedLookupTableEintOp currentOperation,
                    mlir::PatternRewriter &rewriter) const override {
      auto replacement = fuseBackToBackTableLookups(currentOperation, rewriter);
      if (replacement) {
        CompilationOptions currentCompilationOptions =
            getCurrentCompilationOptions();
        if (currentCompilationOptions.printTluFusing) {
          printTluFusing(currentOperation.getT(),
                         currentOperation->getResult(0), *replacement);
        }

        rewriter.replaceAllUsesWith(currentOperation, *replacement);
        return mlir::success();
      }
      return mlir::failure();
    }
  };
  patterns.add<AfterTluPattern>(context);
}

} // namespace FHELinalg
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.cpp.inc"
