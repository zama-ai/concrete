#include "mlir/IR/TypeUtilities.h"

#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.h"

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
          .dyn_cast_or_null<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have a !HLFHE.eint as the element type of the "
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
          .dyn_cast_or_null<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  if (el1Ty == nullptr) {
    op->emitOpError() << "should have a !HLFHE.eint as the element type of the "
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
          .dyn_cast_or_null<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have a !HLFHE.eint as the element type of the "
                         "tensor of operand #0";
    return mlir::failure();
  }
  auto el1Ty =
      op1Ty.getElementType()
          .dyn_cast_or_null<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  if (el1Ty == nullptr) {
    op->emitOpError() << "should have a !HLFHE.eint as the element type of the "
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
          .dyn_cast_or_null<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  if (el0Ty == nullptr) {
    op->emitOpError() << "should have a !HLFHE.eint as the element type of the "
                         "tensor operand";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace impl

} // namespace OpTrait
} // namespace mlir

namespace mlir {
namespace zamalang {
namespace HLFHELinalg {

mlir::LogicalResult verifyApplyLookupTable(ApplyLookupTableEintOp &op) {
  auto tTy = op.t().getType().cast<mlir::RankedTensorType>();
  auto tEltTy =
      tTy.getElementType().cast<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  auto lutTy = op.lut().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();
  auto resultTy = op.getResult().getType().cast<mlir::RankedTensorType>();

  // Check the shape of lut argument
  auto tEltwidth = tEltTy.getWidth();
  mlir::SmallVector<int64_t, 1> expectedShape{1 << tEltwidth};
  if (!lutTy.hasStaticShape(expectedShape) || !lutEltTy.isInteger(64)) {
    op.emitOpError()
        << "should have as operand #2 a tensor<2^pxi64>, where p is the width "
           "of the encrypted integer of the operand #1,"
        << "expect tensor <" << expectedShape[0] << "xi64>";
    return mlir::failure();
  }
  if (!resultTy.hasStaticShape(tTy.getShape())) {
    op.emitOpError()
        << " should have same shapes for operand #1 and the result";
  }
  return mlir::success();
}

mlir::LogicalResult
verifyApplyMultiLookupTable(ApplyMultiLookupTableEintOp &op) {
  auto tTy = op.t().getType().cast<mlir::RankedTensorType>();
  auto tEltTy =
      tTy.getElementType().cast<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  auto lutTy = op.luts().getType().cast<mlir::RankedTensorType>();
  auto lutEltTy = lutTy.getElementType().cast<mlir::IntegerType>();
  auto resultTy = op.getResult().getType().cast<mlir::RankedTensorType>();

  // Check the shape of luts argument
  auto lut_size = lutTy.getShape()[lutTy.getShape().size() - 1];
  auto expected_lut_size = 1 << tEltTy.getWidth();
  if (lut_size != expected_lut_size || !lutEltTy.isInteger(64)) {
    op.emitOpError() << "should have as operand #2 a "
                        "tensor<DMx...xD1X2^pxi64>, where p is the width "
                        "of the encrypted integer of the operand #1,"
                     << "expect tensor <DMx...xD1X" << expected_lut_size
                     << "xi64>";
    return mlir::failure();
  }
  if (!resultTy.hasStaticShape(tTy.getShape())) {
    op.emitOpError()
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

HLFHE::EncryptedIntegerType getEncryptedElmentType(::mlir::Value value) {
  using namespace mlir::zamalang::HLFHE;
  return getElmentType<HLFHE::EncryptedIntegerType>(value);
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
  HLFHE::emitErrorBadLutSize(op, "luts", "ct", expected_lut_size, index_width);
  return mlir::failure();
}

mlir::LogicalResult
verifyApplyMappedLookupTable(ApplyMappedLookupTableEintOp &op) {
  auto t = op.t();
  auto luts = op.luts();
  auto map = op.map();
  auto result = op.getResult();

  auto t_shape = getTensorType(t).getShape();
  if (!getTensorType(result).hasStaticShape(t_shape)) {
    op.emitOpError()
        << ": `t` (operand #1) and `map` (operand #2) must have the same shape";
    return mlir::failure();
  }

  if (!getTensorType(map).getElementType().isIndex()) {
    op.emitOpError()
        << ": `map` (operand #3) should contains elements of type `index`";
    return mlir::failure();
  }

  return mlir::success(verifyMapHasRightShape(op, t, map).succeeded() &&
                       verifyLutsSize(op, t, luts).succeeded());
}

::mlir::LogicalResult verifyDotEintInt(Dot &op) {
  if (::mlir::failed(mlir::verifyCompatibleShape(op.lhs().getType(),
                                                 op.rhs().getType()))) {
    return op.emitOpError("arguments have incompatible shapes");
  }
  auto lhsEltType = op.lhs()
                        .getType()
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<HLFHE::EncryptedIntegerType>();
  auto rhsEltType = op.rhs()
                        .getType()
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<mlir::IntegerType>();
  auto resultType =
      op.getResult().getType().cast<HLFHE::EncryptedIntegerType>();
  if (!mlir::zamalang::HLFHE::verifyEncryptedIntegerAndIntegerInputsConsistency(
          op, lhsEltType, rhsEltType)) {
    return ::mlir::failure();
  }
  if (!HLFHE::verifyEncryptedIntegerInputAndResultConsistency(op, lhsEltType,
                                                              resultType)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

/// Verify the matmul shapes, the type of tensor elements should be checked by
/// something else
template <typename MatMulOp> mlir::LogicalResult verifyMatmul(MatMulOp &op) {
  auto lhsTy = ((mlir::Type)op.lhs().getType()).cast<mlir::RankedTensorType>();

  auto rhsTy = ((mlir::Type)op.rhs().getType()).cast<mlir::RankedTensorType>();

  auto resultTy =
      ((mlir::Type)op.getResult().getType()).cast<mlir::RankedTensorType>();

  if (lhsTy.getShape().size() != 2 || rhsTy.getShape().size() != 2) {
    op.emitOpError() << "should have 2D tensors as operands";
    return mlir::failure();
  }
  if (lhsTy.getDimSize(1) != rhsTy.getDimSize(0)) {
    op.emitOpError() << "should have the dimension #0 of operand #1"
                        "equals to the dimension #1 of operand #0, expect "
                     << lhsTy.getDimSize(1) << " got " << rhsTy.getDimSize(0);
    return mlir::failure();
  }
  // Check the shape of lut argument
  mlir::SmallVector<int64_t, 2> expectedShape{lhsTy.getDimSize(0),
                                              rhsTy.getDimSize(1)};
  if (!resultTy.hasStaticShape(expectedShape)) {
    op.emitOpError() << "should have the result shape compatible with operands "
                     << "shape, expect " << expectedShape[0] << "x"
                     << expectedShape[1] << " as the shape of the result";
    return mlir::failure();
  }
  return mlir::success();
}
} // namespace HLFHELinalg
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.cpp.inc"
