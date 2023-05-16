// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include <concretelang/Dialect/FHE/Analysis/MANP.h>
#include <concretelang/Dialect/FHE/Analysis/utils.h>
#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h>
#include <concretelang/Support/math.h>
#include <mlir/IR/BuiltinOps.h>

#include <limits>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallString.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/FHE/Analysis/MANP.h.inc>

namespace mlir {
namespace concretelang {
namespace {

/// Returns `true` if the given value is a scalar or tensor argument of
/// a function, for which a MANP of 1 can be assumed.
static bool isEncryptedFunctionParameter(mlir::Value value) {
  if (!value.isa<mlir::BlockArgument>())
    return false;

  mlir::Block *block = value.cast<mlir::BlockArgument>().getOwner();

  if (!block || !block->getParentOp() ||
      !llvm::isa<mlir::func::FuncOp>(block->getParentOp())) {
    return false;
  }

  return mlir::concretelang::fhe::utils::isEncryptedValue(value);
}

/// The `MANPLatticeValue` represents the squared Minimal Arithmetic
/// Noise Padding for an operation using the squared 2-norm of an
/// equivalent dot operation. This can either be an actual value if the
/// values for its predecessors have been calculated beforehand or an
/// unknown value otherwise.
struct MANPLatticeValue {
  MANPLatticeValue(std::optional<llvm::APInt> manp = {}) : manp(manp) {}

  static MANPLatticeValue getPessimisticValueState(mlir::MLIRContext *context) {
    return MANPLatticeValue();
  }

  static MANPLatticeValue getPessimisticValueState(mlir::Value value) {
    // Function arguments are assumed to require a Minimal Arithmetic
    // Noise Padding with a 2-norm of 1.
    //
    // TODO: Provide a mechanism to propagate Minimal Arithmetic Noise
    // Padding across function calls.
    if (isEncryptedFunctionParameter(value)) {
      return MANPLatticeValue(llvm::APInt{1, 1, false});
    } else {
      // All other operations have an unknown Minimal Arithmetic Noise
      // Padding until an value for all predecessors has been
      // calculated.
      return MANPLatticeValue();
    }
  }

  bool operator==(const MANPLatticeValue &rhs) const {
    return this->manp == rhs.manp;
  }

  static MANPLatticeValue join(const MANPLatticeValue &lhs,
                               const MANPLatticeValue &rhs) {
    if (!lhs.getMANP().has_value())
      return rhs;
    if (!rhs.getMANP().has_value())
      return lhs;

    if (lhs.getMANP().value() == rhs.getMANP().value())
      return lhs;

    assert(false && "Attempting to join two distinct initialized values");
    return MANPLatticeValue{};
  }

  void print(raw_ostream &os) const {
    if (manp.has_value())
      os << manp.value();
    else
      os << "(undefined)";
  }

  std::optional<llvm::APInt> getMANP() const { return manp; }

protected:
  std::optional<llvm::APInt> manp;
};

class MANPLattice : public mlir::dataflow::Lattice<MANPLatticeValue> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MANPLattice)

  using Lattice::Lattice;
};

/// Checks if `lhs` is less than `rhs`, where both values are assumed
/// to be positive. The bit width of the smaller `APInt` is extended
/// before comparison via `APInt::ult`.
static bool APIntWidthExtendULT(const llvm::APInt &lhs,
                                const llvm::APInt &rhs) {
  if (lhs.getBitWidth() < rhs.getBitWidth())
    return lhs.zext(rhs.getBitWidth()).ult(rhs);
  else if (lhs.getBitWidth() > rhs.getBitWidth())
    return lhs.ult(rhs.zext(lhs.getBitWidth()));
  else
    return lhs.ult(rhs);
}

/// Adds two `APInt` values, where both values are assumed to be
/// positive. The bit width of the operands is extended in order to
/// guarantee that the sum fits into the resulting `APInt`.
static llvm::APInt APIntWidthExtendUAdd(const llvm::APInt &lhs,
                                        const llvm::APInt &rhs) {
  unsigned maxBits = std::max(lhs.getBitWidth(), rhs.getBitWidth());

  // Make sure the required number of bits can be represented by the
  // `unsigned` argument of `zext`.
  assert(std::numeric_limits<unsigned>::max() - maxBits > 1);

  unsigned targetWidth = maxBits + 1;
  return lhs.zext(targetWidth) + rhs.zext(targetWidth);
}

/// Multiplies two `APInt` values, where both values are assumed to be
/// positive. The bit width of the operands is extended in order to
/// guarantee that the product fits into the resulting `APInt`.
static llvm::APInt APIntWidthExtendUMul(const llvm::APInt &lhs,
                                        const llvm::APInt &rhs) {
  // Make sure the required number of bits can be represented by the
  // `unsigned` argument of `zext`.
  assert(std::numeric_limits<unsigned>::max() -
                 std::max(lhs.getBitWidth(), rhs.getBitWidth()) >
             std::min(lhs.getBitWidth(), rhs.getBitWidth()) &&
         "Required number of bits cannot be represented with an APInt");

  unsigned targetWidth = lhs.getBitWidth() + rhs.getBitWidth();
  return lhs.zext(targetWidth) * rhs.zext(targetWidth);
}

/// Returns the maximum value beetwen `lhs` and `rhs`, where both values are
/// assumed to be positive. The bit width of the smaller `APInt` is extended
/// before comparison via `APInt::ult`.
static llvm::APInt APIntUMax(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  if (APIntWidthExtendULT(lhs, rhs)) {
    return rhs;
  }
  return lhs;
}

/// Calculates the square of `i`. The bit width `i` is extended in
/// order to guarantee that the product fits into the resulting
/// `APInt`.
static llvm::APInt APIntWidthExtendUnsignedSq(const llvm::APInt &i) {
  // Make sure the required number of bits can be represented by the
  // `unsigned` argument of `zext`.
  assert(i.getBitWidth() < std::numeric_limits<unsigned>::max() / 2 &&
         "Required number of bits cannot be represented with an APInt");
  llvm::APInt ie = i.zext(2 * i.getBitWidth());

  return ie * ie;
}

/// Calculates the square of the value of `i`.
static llvm::APInt APIntWidthExtendSqForConstant(const llvm::APInt &i) {
  auto extI = i.sext(2 * i.getBitWidth());
  return extI * extI;
}

/// Calculates the square root of `i` and rounds it to the next highest
/// integer value (i.e., the square of the result is guaranteed to be
/// greater or equal to `i`).
static llvm::APInt APIntCeilSqrt(const llvm::APInt &i) {
  llvm::APInt res = i.sqrt();
  llvm::APInt resSq = APIntWidthExtendUnsignedSq(res);

  if (APIntWidthExtendULT(resSq, i))
    return APIntWidthExtendUAdd(res, llvm::APInt{1, 1, false});
  else
    return res;
}

/// Returns a string representation of `i` assuming that `i` is an
/// unsigned value.
static std::string APIntToStringValUnsigned(const llvm::APInt &i) {
  llvm::SmallString<32> s;
  i.toStringUnsigned(s);
  return std::string(s.c_str());
}

/// Returns the squared 2-norm for a dynamic integer by conservatively
/// assuming that the integer's value is the maximum for the integer
/// width.
static llvm::APInt conservativeIntNorm2Sq(mlir::Type t) {
  assert(t.isSignlessInteger() && "Type must be a signless integer type");
  assert(std::numeric_limits<unsigned>::max() - t.getIntOrFloatBitWidth() > 1);

  // we consider the maximum value as a signed integer
  llvm::APInt maxVal = APInt::getMaxValue(t.getIntOrFloatBitWidth() - 1);
  return APIntWidthExtendUnsignedSq(maxVal);
}

static llvm::APInt
getNoOpSqMANP(llvm::ArrayRef<const MANPLattice *> operandMANPs) {
  // Come from block arg as example
  if (operandMANPs.size() == 0) {
    return llvm::APInt{1, 1, false};
  }
  assert(operandMANPs[0]->getValue().getMANP().has_value() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().value();
  return eNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of an
/// `FHELinalg.dot_eint_eint` operation.
static llvm::APInt getSqMANP(mlir::concretelang::FHELinalg::DotEint op,
                             llvm::ArrayRef<const MANPLattice *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().has_value() &&
         operandMANPs[1]->getValue().getMANP().has_value() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt lhsNorm = operandMANPs[0]->getValue().getMANP().value();
  llvm::APInt rhsNorm = operandMANPs[1]->getValue().getMANP().value();

  auto rhsType =
      ((mlir::Type)op.getRhs().getType()).cast<mlir::RankedTensorType>();

  llvm::ArrayRef<int64_t> rhsShape = rhsType.getShape();

  int64_t rhsDims = (int64_t)rhsShape.size();

  assert(rhsDims == 1 && "In MANP computation dot product RHS expected to have "
                         "a single dimension");

  int64_t N = rhsShape[0];

  // Compute output MANP:
  // Tlu output MANP is 1
  llvm::APInt tlu = {1, 1, false};
  // The element-wise multiplication is given by the
  // subtraction of two TLU outputs. The MANP of the multiplication is thus
  // the sum of the TLU MANPs
  llvm::APInt elemMulNorm = APIntWidthExtendUAdd(tlu, tlu);

  llvm::APInt accNorm = llvm::APInt{1, 0, false};

  return APIntWidthExtendUMul(APIntWidthExtendUAdd(tlu, tlu),
                              llvm::APInt(ceilLog2(N + 1), N, false));
}

/// Calculates the squared Minimal Arithmetic Noise Padding of an unary FHE
/// operation.
static std::optional<llvm::APInt>
getSqMANP(mlir::concretelang::FHE::UnaryEint op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {
  // not all unary ops taking an encrypted operand have a type signature
  // reflecting that, a check might be required (FHELinalg.TransposeOp is one
  // such known op)
  if (op.operandIntType().isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
    assert(operandMANPs.size() == 1 &&
           operandMANPs[0]->getValue().getMANP().has_value() &&
           "Missing squared Minimal Arithmetic Noise Padding for encrypted "
           "operand");
    return op.sqMANP(operandMANPs[0]->getValue().getMANP().value());
  } else
    return {};
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a binary FHE
/// operation with the first operand encrypted.

static std::optional<llvm::APInt>
getSqMANP(mlir::concretelang::FHE::BinaryEintInt op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {
  assert(operandMANPs.size() >= 2 && // conv2d has an optional 3rd operand
         operandMANPs[0]->getValue().getMANP().has_value() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operand");
  return op.sqMANP(operandMANPs[0]->getValue().getMANP().value());
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a binary FHE
/// operation with the second operand encrypted.

static std::optional<llvm::APInt>
getSqMANP(mlir::concretelang::FHE::BinaryIntEint op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {
  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");
  return op.sqMANP(operandMANPs[1]->getValue().getMANP().value());
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a binary FHE
/// operation with both operands encrypted.

static std::optional<llvm::APInt>
getSqMANP(mlir::concretelang::FHE::BinaryEint op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().has_value() &&
         operandMANPs[1]->getValue().getMANP().has_value() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  return op.sqMANP(operandMANPs[0]->getValue().getMANP().value(),
                   operandMANPs[1]->getValue().getMANP().value());
}

static llvm::APInt sqMANP_mul_eint_int(llvm::APInt a, mlir::Type iTy,
                                       std::optional<llvm::APInt> b) {
  assert(iTy.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  llvm::APInt sqNorm;
  if (b.has_value()) {
    // For a constant operand use actual constant to calculate 2-norm
    sqNorm = APIntWidthExtendSqForConstant(b.value());
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUMul(sqNorm, a);
}

static llvm::APInt sqMANP_mul_eint(llvm::APInt a, llvm::APInt b) {
  // a * b = ((a + b)^2 / 4) - ((a - b)^2 / 4) == tlu(a + b) - tlu(a - b)
  const llvm::APInt beforeTLUs = APIntWidthExtendUAdd(a, b);
  const llvm::APInt tlu = {1, 1, false};
  const llvm::APInt result = APIntWidthExtendUAdd(tlu, tlu);

  return result;
}

/// Computes the squared vector norm as the maximum of all dot products over the
/// destroyed dimension. The computation is recursive and can be seen as folding
/// a tree of values where leaves compute the dot products and nodes choose the
/// maximum of the children. There is a branching node at every shape dimension
/// with the fanout equal to that dimension range, except for the destroyed
/// dimension (where the fanout is 1).
///
/// This function is expected to behave correctly on clear tensor shapes with
/// any dimensionality, for example:
///
///     MxN @ N ->     M
///   LxMxN @ N ->   LxM
/// KxLxMxN @ N -> KxLxM
///
/// N @     NxP ->     P
/// N @   LxNxP ->   LxP
/// N @ KxLxNxP -> KxLxP
///
/// KxLxMxN @   NxP -> KxLxMxP
/// KxLxMxN @ LxNxP -> KxLxMxP
/// Kx1xMxN @ LxNxP -> KxLxMxP
///
///   MxN @ KxLxNxP -> KxLxMxP
/// LxMxN @ KxLxNxP -> KxLxMxP
/// 1xMxN @ KxLxNxP -> KxLxMxP
///
/// N @     NxP ->     P
/// N @   LxNxP ->   LxP
/// N @ KxLxNxP -> KxLxP
///
///     MxN @ N ->     M
///   LxMxN @ N ->   LxM
/// KxLxMxN @ N -> KxLxM
///
/// MxN @ NxP -> MxP

static llvm::APInt sqMANP_matmul_internal(
    llvm::ArrayRef<int64_t> shape, size_t destroyedDimension,
    llvm::SmallVector<uint64_t, /*size-hint=*/4> iterPoint,
    mlir::detail::ElementsAttrRange<mlir::DenseElementsAttr::IntElementIterator>
        clearValues,
    llvm::APInt encryptedOperandNorm) {
  assert(iterPoint.size() >= shape.size() &&
         "Tensor shape dimensionality is larger than iteration space "
         "dimensionality");
  assert(destroyedDimension < iterPoint.size() &&
         "Destroyed dimension outside of iteration space dimensionality");
  size_t currentDimension = iterPoint.size() - shape.size();

  if (currentDimension == destroyedDimension) {
    // the dot product over destroyed dimension will sum products counting down
    // from the largest index
    iterPoint[currentDimension] = shape[0] - 1;
    return sqMANP_matmul_internal(shape.drop_front(1), destroyedDimension,
                                  iterPoint, clearValues, encryptedOperandNorm);
  }

  if (shape.size() == 0) { // `iterPoint` is defined in all indices, let's
                           // compute the dot product
    llvm::APInt accumulationNorm = llvm::APInt{1, 0, false};
    for (int64_t i = iterPoint[destroyedDimension]; i >= 0; i--) {
      iterPoint[destroyedDimension] = i;

      llvm::APInt weight = clearValues[iterPoint];
      llvm::APInt weightNorm = APIntWidthExtendSqForConstant(weight);
      llvm::APInt multiplicationNorm =
          APIntWidthExtendUMul(encryptedOperandNorm, weightNorm);
      accumulationNorm =
          APIntWidthExtendUAdd(multiplicationNorm, accumulationNorm);
    }
    return accumulationNorm;
  } else { // descend into all indices in current dimension
    llvm::APInt maximumNorm = llvm::APInt{1, 1, false};
    for (int64_t i = 0; i < shape[0]; i++) {
      iterPoint[currentDimension] = i;
      llvm::APInt accumulationNorm =
          sqMANP_matmul_internal(shape.drop_front(1), destroyedDimension,
                                 iterPoint, clearValues, encryptedOperandNorm);
      maximumNorm = APIntUMax(maximumNorm, accumulationNorm);
    }
    return maximumNorm;
  }
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.mul_eint_int` operation.
static llvm::APInt
sqMANP_matmul(llvm::APInt encryptedOperandNorm,
              mlir::RankedTensorType clearOperandType,
              std::optional<mlir::detail::ElementsAttrRange<
                  mlir::DenseElementsAttr::IntElementIterator>>
                  clearVals,
              unsigned clearOpNum) {

  assert(clearOperandType.getElementType().isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  llvm::ArrayRef<int64_t> clearOperandShape = clearOperandType.getShape();
  uint64_t clearOperandDims = (uint64_t)clearOperandShape.size();
  // if the clear operand is LHS (index 0), then the destroyed dimension is its
  // last (dims-1) if the clear operand is RHS (index 1), then the destroyed
  // dimension is its second to last (dims-2)
  assert(clearOpNum <= 1 && "Cannot determine destroyed dimension: operation "
                            "has more than 2 operands");
  size_t destroyedDimension =
      clearOperandDims == 1 ? 0 : clearOperandDims - 1 - clearOpNum;

  llvm::APInt accNorm = llvm::APInt{1, 0, false};

  if (clearVals.has_value())
    accNorm =
        sqMANP_matmul_internal(clearOperandShape, destroyedDimension,
                               llvm::SmallVector<uint64_t, /*size-hint=*/4>(
                                   clearOperandShape.size(), 0),
                               clearVals.value(), encryptedOperandNorm);
  else {
    llvm::APInt clearOperandNorm =
        conservativeIntNorm2Sq(clearOperandType.getElementType());
    llvm::APInt mulNorm =
        APIntWidthExtendUMul(encryptedOperandNorm, clearOperandNorm);
    uint64_t N = clearOperandShape[destroyedDimension];
    unsigned int Nbits = ceilLog2(N + 1);
    mulNorm = APIntWidthExtendUMul(mulNorm, APInt{Nbits, N, false});
    accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
  }

  return accNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a matmul
/// operation
static llvm::APInt getSqMANP(mlir::concretelang::FHELinalg::MatMulEintEintOp op,
                             llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  auto rhsType =
      ((mlir::Type)op.getRhs().getType()).cast<mlir::RankedTensorType>();

  llvm::ArrayRef<int64_t> rhsShape = rhsType.getShape();

  int64_t rhsDims = (int64_t)rhsShape.size();

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt lhsNorm = operandMANPs[0]->getValue().getMANP().value();
  llvm::APInt rhsNorm = operandMANPs[1]->getValue().getMANP().value();

  int64_t N = rhsDims <= 2 ? rhsShape[0] : rhsShape[rhsDims - 2];

  // Compute MANP of a single matrix cell x matrix cell multiplication
  // This is used later to compute the MANP of an entire dot product

  llvm::APInt tlu = {1, 1, false};
  llvm::APInt elemMulNorm = APIntWidthExtendUAdd(tlu, tlu);
  llvm::APInt accNorm = llvm::APInt{1, 0, false};

  // For the total MatMul MANP, take the MANP of a single
  // column-row dot-product
  // All such dot-products produce the same MANP, there
  // is no need to take the maximum over the dot-products
  for (int64_t i = 0; i < N; i++) {
    accNorm = APIntWidthExtendUAdd(elemMulNorm, accNorm);
  }

  return accNorm;
}

static llvm::APInt getSqMANP(mlir::tensor::ExtractOp op,
                             llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  assert(
      operandMANPs[0]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().value();

  return eNorm;
}

static std::optional<llvm::APInt>
getSqMANP(mlir::tensor::FromElementsOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  auto max = std::max_element(operandMANPs.begin(), operandMANPs.end(),
                              [](const MANPLattice *a, const MANPLattice *b) {
                                return APIntWidthExtendULT(
                                    a->getValue().getMANP().value(),
                                    b->getValue().getMANP().value());
                              });
  return (*max)->getValue().getMANP().value();
}

static std::optional<llvm::APInt>
getSqMANP(mlir::tensor::ExtractSliceOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  assert(
      operandMANPs[0]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().value();
}

static std::optional<llvm::APInt>
getSqMANP(mlir::tensor::InsertSliceOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  assert(
      operandMANPs.size() >= 2 &&
      operandMANPs[0]->getValue().getMANP().has_value() &&
      operandMANPs[1]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return APIntUMax(operandMANPs[0]->getValue().getMANP().value(),
                   operandMANPs[1]->getValue().getMANP().value());
}

static std::optional<llvm::APInt>
getSqMANP(mlir::tensor::InsertOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  assert(
      operandMANPs.size() >= 2 &&
      operandMANPs[0]->getValue().getMANP().has_value() &&
      operandMANPs[1]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return APIntUMax(operandMANPs[0]->getValue().getMANP().value(),
                   operandMANPs[1]->getValue().getMANP().value());
}

static std::optional<llvm::APInt>
getSqMANP(mlir::tensor::CollapseShapeOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  assert(
      operandMANPs.size() >= 1 &&
      operandMANPs[0]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().value();
}

static std::optional<llvm::APInt>
getSqMANP(mlir::tensor::ExpandShapeOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  assert(
      operandMANPs.size() >= 1 &&
      operandMANPs[0]->getValue().getMANP().has_value() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().value();
}

static std::optional<llvm::APInt>
getSqMANP(mlir::concretelang::FHELinalg::SumOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  auto inputType = op.getOperand().getType().dyn_cast<mlir::TensorType>();

  uint64_t numberOfElementsInTheInput = inputType.getNumElements();
  if (numberOfElementsInTheInput == 0) {
    return llvm::APInt{1, 1, false};
  }

  uint64_t numberOfElementsAddedTogetherInEachOutputCell = 1;

  mlir::ArrayAttr axes = op.getAxes();
  if (axes.empty()) {
    numberOfElementsAddedTogetherInEachOutputCell *= numberOfElementsInTheInput;
  } else {
    llvm::ArrayRef<int64_t> shape = inputType.getShape();
    for (mlir::Attribute axisAttribute : op.getAxes()) {
      int64_t axis = axisAttribute.cast<IntegerAttr>().getInt();
      numberOfElementsAddedTogetherInEachOutputCell *= shape[axis];
    }
  }

  unsigned int noiseMultiplierBits =
      ceilLog2(numberOfElementsAddedTogetherInEachOutputCell + 1);

  auto noiseMultiplier = llvm::APInt{
      noiseMultiplierBits,
      numberOfElementsAddedTogetherInEachOutputCell,
      false,
  };

  assert(operandMANPs.size() == 1 &&
         operandMANPs[0]->getValue().getMANP().has_value() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt operandMANP = operandMANPs[0]->getValue().getMANP().value();

  return APIntWidthExtendUMul(noiseMultiplier, operandMANP);
}

static std::optional<llvm::APInt>
getSqMANP(mlir::concretelang::FHELinalg::ConcatOp op,
          llvm::ArrayRef<const MANPLattice *> operandMANPs) {

  llvm::APInt result = llvm::APInt{1, 0, false};
  for (const MANPLattice *operandMANP : operandMANPs) {
    llvm::APInt candidate = operandMANP->getValue().getMANP().value();
    if (candidate.getLimitedValue() >= result.getLimitedValue()) {
      result = candidate;
    }
  }
  return result;
}

static llvm::APInt
sqMANP_conv2d(llvm::APInt inputNorm, mlir::RankedTensorType weightTy,
              std::optional<mlir::detail::ElementsAttrRange<
                  mlir::DenseElementsAttr::IntElementIterator>>
                  weightVals) {
  // Initial value of the accumulator to 0
  llvm::APInt accNorm = llvm::APInt{1, 0, false};

  // Weight shapes: Filter*Channel*Height*Width
  uint64_t F = weightTy.getShape()[0];
  uint64_t C = weightTy.getShape()[1];
  uint64_t H = weightTy.getShape()[2];
  uint64_t W = weightTy.getShape()[3];
  if (weightVals.has_value()) {
    // For a constant weight kernel use actual constant to calculate 2-norm
    // input windows are being multiplied by a kernel and summed up
    for (uint64_t f = 0; f < F; f++) {
      llvm::APInt tmpNorm = inputNorm;

      for (uint64_t c = 0; c < C; c++) {
        for (uint64_t h = 0; h < H; h++) {
          for (uint64_t w = 0; w < W; w++) {
            llvm::APInt cst = weightVals.value()[{f, c, h, w}];
            llvm::APInt weightNorm = APIntWidthExtendSqForConstant(cst);
            llvm::APInt mulNorm = APIntWidthExtendUMul(inputNorm, weightNorm);
            tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
          }
        }
      }
      // Take the max of the 2-norm on the filter
      accNorm = APIntUMax(accNorm, tmpNorm);
    }
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    llvm::APInt weightNorm = conservativeIntNorm2Sq(weightTy.getElementType());
    // For a weight (kernel) of shape tensor<FxCxHxW>, there is C*H*W
    // FHE.mul_eint_int and FHE.add_eint operations for each elements of the
    // result
    int64_t n_mul = C * H * W;
    llvm::APInt mulNorm = APIntWidthExtendUMul(inputNorm, weightNorm);
    for (int64_t i = 0; i < n_mul; i++) {
      accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
    }
  }
  return accNorm;
}

class MANPAnalysis
    : public mlir::dataflow::SparseDataFlowAnalysis<MANPLattice> {
public:
  explicit MANPAnalysis(mlir::DataFlowSolver &solver, bool debug)
      : mlir::dataflow::SparseDataFlowAnalysis<MANPLattice>(solver),
        debug(debug) {}

  void setToEntryState(MANPLattice *lattice) override {
    if (isEncryptedFunctionParameter(lattice->getPoint())) {
      // Set minimal MANP for encrypted function arguments
      propagateIfChanged(lattice, lattice->join(MANPLatticeValue{
                                      std::optional{llvm::APInt(1, 1)}}));
    } else {
      // Everything else is initialized with an unset value
      propagateIfChanged(lattice, lattice->join(MANPLatticeValue{}));
    }
  }

  void visitOperation(Operation *op, ArrayRef<const MANPLattice *> operands,
                      ArrayRef<MANPLattice *> results) override {
    MANPLattice *latticeRes = results[0];

    std::optional<llvm::APInt> norm2SqEquiv;

    if (auto cstNoiseOp =
            llvm::dyn_cast<mlir::concretelang::FHE::ConstantNoise>(op)) {
      norm2SqEquiv = llvm::APInt{1, 1, false};
    } else if (llvm::isa<mlir::concretelang::FHE::ToBoolOp>(op) ||
               llvm::isa<mlir::concretelang::FHE::FromBoolOp>(op)) {
      norm2SqEquiv = getNoOpSqMANP(operands);
    }
    // FHE and FHELinalg Operators
    else if (auto unaryEintOp =
                 llvm::dyn_cast<mlir::concretelang::FHE::UnaryEint>(op)) {
      norm2SqEquiv = getSqMANP(unaryEintOp, operands);
    } else if (auto binaryEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::BinaryEintInt>(op)) {
      norm2SqEquiv = getSqMANP(binaryEintIntOp, operands);
    } else if (auto binaryIntEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::BinaryIntEint>(op)) {
      norm2SqEquiv = getSqMANP(binaryIntEintOp, operands);
    } else if (auto binaryEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::BinaryEint>(op)) {
      norm2SqEquiv = getSqMANP(binaryEintOp, operands);
    } else if (auto dotEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::DotEint>(op)) {
      norm2SqEquiv = getSqMANP(dotEintOp, operands);
    } else if (auto matmulEintEintOp = llvm::dyn_cast<
                   mlir::concretelang::FHELinalg::MatMulEintEintOp>(op)) {
      norm2SqEquiv = getSqMANP(matmulEintEintOp, operands);
    } else if (auto sumOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::SumOp>(op)) {
      norm2SqEquiv = getSqMANP(sumOp, operands);
    } else if (auto concatOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::ConcatOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(concatOp, operands);
    } else if (auto fromElementOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::FromElementOp>(
                       op)) {
      if (operands[0]->getValue().getMANP().has_value()) {
        norm2SqEquiv = operands[0]->getValue().getMANP().value();
      } else
        norm2SqEquiv = llvm::APInt{1, 1, false};
    }
    // Tensor Operators
    // ExtractOp
    else if (auto extractOp = llvm::dyn_cast<mlir::tensor::ExtractOp>(op)) {
      if (extractOp.getResult()
              .getType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(extractOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }
    // ExtractSliceOp
    else if (auto extractSliceOp =
                 llvm::dyn_cast<mlir::tensor::ExtractSliceOp>(op)) {
      if (extractSliceOp.getResult()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(extractSliceOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }
    // InsertOp
    else if (auto insertOp = llvm::dyn_cast<mlir::tensor::InsertOp>(op)) {
      if (insertOp.getResult()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(insertOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }
    // InsertSliceOp
    else if (auto insertSliceOp =
                 llvm::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
      if (insertSliceOp.getResult()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(insertSliceOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }
    // FromElementOp
    else if (auto fromOp = llvm::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
      if (fromOp.getResult()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(fromOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }
    // TensorCollapseShapeOp
    else if (auto reshapeOp =
                 llvm::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
      if (reshapeOp.getResult()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(reshapeOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }
    // TensorExpandShapeOp
    else if (auto reshapeOp = llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
      if (reshapeOp.getResult()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::FheIntegerInterface>()) {
        norm2SqEquiv = getSqMANP(reshapeOp, operands);
      } else {
        norm2SqEquiv = {};
      }
    }

    else if (llvm::isa<mlir::arith::ConstantOp>(op)) {
      norm2SqEquiv = {};
    } else if (llvm::isa<mlir::concretelang::FHE::FHEDialect>(
                   *op->getDialect())) {
      op->emitError("Unsupported operation");
      assert(false && "Unsupported operation");
    } else {
      norm2SqEquiv = {};
    }

    if (norm2SqEquiv.has_value()) {
      latticeRes->join(MANPLatticeValue{norm2SqEquiv});

      op->setAttr("SMANP",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(
                          op->getContext(), norm2SqEquiv.value().getBitWidth(),
                          mlir::IntegerType::SignednessSemantics::Unsigned),
                      norm2SqEquiv.value()));

      llvm::APInt norm2Equiv = APIntCeilSqrt(norm2SqEquiv.value());

      op->setAttr("MANP",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(
                          op->getContext(), norm2Equiv.getBitWidth(),
                          mlir::IntegerType::SignednessSemantics::Unsigned),
                      norm2Equiv));

      if (debug) {
        op->emitRemark("Squared Minimal Arithmetic Noise Padding: ")
            << APIntToStringValUnsigned(norm2SqEquiv.value()) << "\n";
      }
    } else {
      latticeRes->join(MANPLatticeValue{});
    }
  }

private:
  bool debug;
};
} // namespace

namespace FHE {
llvm::APInt AddEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  return APIntWidthExtendUAdd(a, b);
}

llvm::APInt SubEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  return APIntWidthExtendUAdd(a, b);
}

llvm::APInt MulEintIntOp::sqMANP(llvm::APInt a) {
  return sqMANP_mul_eint_int(
      a, this->operandIntType(this->getClearOperandNumber()),
      this->operandMaxConstant(this->getClearOperandNumber()));
}

llvm::APInt MulEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  return sqMANP_mul_eint(a, b);
}

llvm::APInt MaxEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  // max(a, b) = max(a - b, 0) + b
  const llvm::APInt sub = APIntWidthExtendUAdd(a, b);
  const llvm::APInt tlu = {1, 1, false};
  const llvm::APInt add = APIntWidthExtendUAdd(tlu, b);

  // this is not optimal as it can increase the resulting noise unnecessarily
  return APIntUMax(add, sub);
}

llvm::APInt RoundEintOp::sqMANP(llvm::APInt a) {
  uint64_t inputWidth =
      this->getOperand().getType().cast<FHE::FheIntegerInterface>().getWidth();
  uint64_t outputWidth =
      this->getResult().getType().cast<FHE::FheIntegerInterface>().getWidth();
  uint64_t clearedBits = inputWidth - outputWidth;

  return a + clearedBits;
}
} // namespace FHE

namespace FHELinalg {
llvm::APInt AddEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  return APIntWidthExtendUAdd(a, b);
}

llvm::APInt SubEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  return APIntWidthExtendUAdd(a, b);
}

llvm::APInt MulEintIntOp::sqMANP(llvm::APInt a) {
  return sqMANP_mul_eint_int(
      a, this->operandIntType(this->getClearOperandNumber()),
      this->operandMaxConstant(this->getClearOperandNumber()));
}

llvm::APInt MulEintOp::sqMANP(llvm::APInt a, llvm::APInt b) {
  return sqMANP_mul_eint(a, b);
}

llvm::APInt Dot::sqMANP(llvm::APInt a) {
  unsigned clearOpNum = this->getClearOperandNumber();
  auto clearOperandType = this->getOperation()
                              ->getOpOperand(clearOpNum)
                              .get()
                              .getType()
                              .cast<mlir::RankedTensorType>();
  return sqMANP_matmul(a, clearOperandType, this->opTensorConstant(clearOpNum),
                       clearOpNum);
}

llvm::APInt MatMulEintIntOp::sqMANP(llvm::APInt a) {
  unsigned clearOpNum = this->getClearOperandNumber();
  auto clearOpType = this->getOperation()
                         ->getOpOperand(clearOpNum)
                         .get()
                         .getType()
                         .cast<mlir::RankedTensorType>();
  return sqMANP_matmul(a, clearOpType, this->opTensorConstant(clearOpNum),
                       clearOpNum);
}

llvm::APInt MatMulIntEintOp::sqMANP(llvm::APInt a) {
  unsigned clearOpNum = this->getClearOperandNumber();
  assert(clearOpNum <= 1 && "Operation has more than 2 operands");
  auto clearOpType = this->getOperation()
                         ->getOpOperand(clearOpNum)
                         .get()
                         .getType()
                         .cast<mlir::RankedTensorType>();
  return sqMANP_matmul(a, clearOpType, this->opTensorConstant(clearOpNum),
                       clearOpNum);
}

llvm::APInt Conv2dOp::sqMANP(llvm::APInt a) {
  unsigned clearOpNum = this->getClearOperandNumber();
  auto clearOpType = this->getOperation()
                         ->getOpOperand(clearOpNum)
                         .get()
                         .getType()
                         .cast<mlir::RankedTensorType>();
  return sqMANP_conv2d(a, clearOpType, this->opTensorConstant(clearOpNum));
}

llvm::APInt Maxpool2dOp::sqMANP(llvm::APInt a) {
  // maximum between two value is calculated using
  // - max(x - y, 0) + y

  // max is calculated with a TLU so MANP is {1, 1, false}
  // y on the other hand comes from the input or from the previous result

  // in the current implementation, it's the input
  // so the resulting MANP is `{1, 1, false} + MANP input`

  const llvm::APInt tlu = {1, 1, false};
  const llvm::APInt forResult = APIntWidthExtendUAdd(tlu, a);
  const llvm::APInt forIntermediate = APIntWidthExtendUAdd(forResult, a);

  return APIntUMax(forIntermediate, forResult);
}

llvm::APInt RoundOp::sqMANP(llvm::APInt a) {
  const uint64_t inputWidth = this->getOperand()
                                  .getType()
                                  .cast<mlir::RankedTensorType>()
                                  .getElementType()
                                  .cast<FHE::FheIntegerInterface>()
                                  .getWidth();

  const uint64_t outputWidth = this->getResult()
                                   .getType()
                                   .cast<mlir::RankedTensorType>()
                                   .getElementType()
                                   .cast<FHE::FheIntegerInterface>()
                                   .getWidth();

  const uint64_t clearedBits = inputWidth - outputWidth;

  return a + clearedBits;
}

} // namespace FHELinalg

namespace {
/// For documentation see MANP.td
struct MANPPass : public MANPBase<MANPPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<MANPAnalysis>(debug);

    if (failed(solver.initializeAndRun(func)))
      return signalPassFailure();
  }
  MANPPass() = delete;
  MANPPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

/// Create an instance of the Minimal Arithmetic Noise Padding analysis
/// pass. If `debug` is true, for each operation, the pass emits a
/// remark containing the squared Minimal Arithmetic Noise Padding of
/// the equivalent dot operation.
std::unique_ptr<mlir::Pass> createMANPPass(bool debug) {
  return std::make_unique<MANPPass>(debug);
}

namespace {
/// For documentation see MANP.td
struct MaxMANPPass : public MaxMANPBase<MaxMANPPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    func.walk(
        [&](mlir::Operation *childOp) { this->processOperation(childOp); });
  }
  MaxMANPPass() = delete;
  MaxMANPPass(std::function<void(const uint64_t, unsigned)> updateMax)
      : updateMax(updateMax){};

protected:
  void processOperation(mlir::Operation *op) {

    // Process all function arguments and use the default value of 1
    // for MANP and the declarend precision
    if (mlir::func::FuncOp func =
            llvm::dyn_cast_or_null<mlir::func::FuncOp>(op)) {
      for (mlir::BlockArgument blockArg : func.getBody().getArguments()) {
        if (isEncryptedFunctionParameter(blockArg)) {
          unsigned int width = fhe::utils::getEintPrecision(blockArg);
          this->updateMax(1, width);
        }
      }
    }

    // Process all results using MANP attribute from MANP pas
    for (mlir::OpResult res : op->getResults()) {
      mlir::concretelang::FHE::FheIntegerInterface eTy =
          res.getType()
              .dyn_cast_or_null<mlir::concretelang::FHE::FheIntegerInterface>();
      if (eTy == nullptr) {
        auto tensorTy = res.getType().dyn_cast_or_null<mlir::TensorType>();
        if (tensorTy != nullptr) {
          eTy = tensorTy.getElementType()
                    .dyn_cast_or_null<
                        mlir::concretelang::FHE::FheIntegerInterface>();
        }
      }

      if (eTy) {
        mlir::IntegerAttr MANP = op->getAttrOfType<mlir::IntegerAttr>("MANP");

        if (!MANP) {
          op->emitError("2-Norm has not been computed");
          this->signalPassFailure();
          return;
        }

        auto manp = MANP.getValue();
        if (!manp.isIntN(64)) {
          op->emitError("2-Norm cannot be reprensented on 64bits");
          this->signalPassFailure();
          return;
        }
        this->updateMax(manp.getZExtValue(), eTy.getWidth());
      }
    }
  }

  std::function<void(const uint64_t, unsigned)> updateMax;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
createMaxMANPPass(std::function<void(const uint64_t, unsigned)> updateMax) {
  return std::make_unique<MaxMANPPass>(updateMax);
}

} // namespace concretelang
} // namespace mlir
