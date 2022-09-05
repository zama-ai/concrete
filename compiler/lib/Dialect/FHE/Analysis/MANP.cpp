// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

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
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
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
  MANPLatticeValue(llvm::Optional<llvm::APInt> manp = {}) : manp(manp) {}

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

  /// Required by `mlir::LatticeElement::join()`, but should never be
  /// invoked, as `MANPAnalysis::visitOperation()` takes care of
  /// combining the squared Minimal Arithmetic Noise Padding of
  /// operands into the Minimal Arithmetic Noise Padding of the result.
  static MANPLatticeValue join(const MANPLatticeValue &lhs,
                               const MANPLatticeValue &rhs) {
    assert(false && "Minimal Arithmetic Noise Padding values can only be "
                    "combined sensibly when the combining operation is known");
    return MANPLatticeValue{};
  }

  llvm::Optional<llvm::APInt> getMANP() { return manp; }

protected:
  llvm::Optional<llvm::APInt> manp;
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
  // Make sure the required number of bits can be represented by the
  // `unsigned` argument of `zext`.
  assert(i.getActiveBits() < 32 &&
         "Square of the constant cannot be represented on 64 bits");
  return llvm::APInt(2 * i.getActiveBits(),
                     i.getZExtValue() * i.getZExtValue());
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

/// Calculates the square of the 2-norm of a tensor initialized with a
/// dense matrix of constant, signless integers. Aborts if the value
/// type or initialization of of `cstOp` is incorrect.
static llvm::APInt denseCstTensorNorm2Sq(mlir::arith::ConstantOp cstOp,
                                         llvm::APInt eNorm) {
  mlir::DenseIntElementsAttr denseVals =
      cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value");

  assert(denseVals && cstOp.getType().isa<mlir::TensorType>() &&
         "Constant must be a tensor initialized with `dense`");

  mlir::TensorType tensorType = cstOp.getType().cast<mlir::TensorType>();

  assert(tensorType.getElementType().isSignlessInteger() &&
         "Can only handle tensors with signless integer elements");

  llvm::APInt accu{1, 0, false};

  for (llvm::APInt val : denseVals.getValues<llvm::APInt>()) {
    llvm::APInt valSqNorm = APIntWidthExtendSqForConstant(val);
    llvm::APInt mulSqNorm = APIntWidthExtendUMul(valSqNorm, eNorm);
    accu = APIntWidthExtendUAdd(accu, mulSqNorm);
  }

  return accu;
}

/// Calculates the square of the 2-norm of a 1D tensor of signless
/// integers by conservatively assuming that the dynamic values are the
/// maximum for the integer width. Aborts if the tensor type `tTy` is
/// incorrect.
static llvm::APInt denseDynTensorNorm2Sq(mlir::TensorType tTy,
                                         llvm::APInt eNorm) {
  assert(tTy && tTy.getElementType().isSignlessInteger() &&
         tTy.hasStaticShape() && tTy.getRank() == 1 &&
         "Plaintext operand must be a statically shaped 1D tensor of integers");

  // Make sure the log2 of the number of elements fits into an
  // unsigned
  assert(std::numeric_limits<unsigned>::max() > 8 * sizeof(uint64_t));

  unsigned elWidth = tTy.getElementTypeBitWidth();

  llvm::APInt maxVal = APInt::getSignedMaxValue(elWidth);
  llvm::APInt maxValSq = APIntWidthExtendUnsignedSq(maxVal);

  llvm::APInt maxMulSqNorm = APIntWidthExtendUMul(maxValSq, eNorm);

  // Calculate number of bits for APInt to store number of elements
  uint64_t nElts = (uint64_t)tTy.getNumElements();
  assert(std::numeric_limits<int64_t>::max() - nElts > 1);
  unsigned nEltsBits = (unsigned)ceilLog2(nElts + 1);

  llvm::APInt nEltsAP{nEltsBits, nElts, false};

  return APIntWidthExtendUMul(maxMulSqNorm, nEltsAP);
}

/// Returns the squared 2-norm of the maximum value of the dense values.
static llvm::APInt maxIntNorm2Sq(mlir::DenseIntElementsAttr denseVals) {
  auto denseValsAP = denseVals.getValues<llvm::APInt>();

  // For a constant operand use actual constant to calculate 2-norm
  llvm::APInt maxCst = denseValsAP[0];
  for (int64_t i = 0; i < denseVals.getNumElements(); i++) {
    llvm::APInt iCst = denseValsAP[i];
    if (maxCst.ult(iCst)) {
      maxCst = iCst;
    }
  }
  return APIntWidthExtendSqForConstant(maxCst);
}

/// Returns the squared 2-norm for a dynamic integer by conservatively
/// assuming that the integer's value is the maximum for the integer
/// width.
static llvm::APInt conservativeIntNorm2Sq(mlir::Type t) {
  assert(t.isSignlessInteger() && "Type must be a signless integer type");
  assert(std::numeric_limits<unsigned>::max() - t.getIntOrFloatBitWidth() > 1);

  llvm::APInt maxVal = APInt::getMaxValue(t.getIntOrFloatBitWidth());
  return APIntWidthExtendUnsignedSq(maxVal);
}

/// Calculates the squared Minimal Arithmetic Noise Padding of an
/// `FHELinalg.dot_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::Dot op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(1).get().getDefiningOp());

  if (cstOp) {
    // Dot product between a vector of encrypted integers and a vector
    // of plaintext constants -> return 2-norm of constant vector
    return denseCstTensorNorm2Sq(cstOp, eNorm);
  } else {
    // Dot product between a vector of encrypted integers and a vector
    // of dynamic plaintext values -> conservatively assume that all
    // the values are the maximum possible value for the integer's
    // width
    mlir::TensorType tTy = op->getOpOperand(1)
                               .get()
                               .getType()
                               .dyn_cast_or_null<mlir::TensorType>();

    return denseDynTensorNorm2Sq(tTy, eNorm);
  }
}

/// Calculates the squared Minimal Arithmetic Noise Padding of an
/// `FHE.add_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::AddEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.add_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::AddEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         operandMANPs[1]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt a = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt b = operandMANPs[1]->getValue().getMANP().getValue();

  return APIntWidthExtendUAdd(a, b);
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.sub_int_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::SubIntEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[1]->getValue().getMANP().getValue();

  return eNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.sub_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::SubEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.sub_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::SubEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         operandMANPs[1]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt a = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt b = operandMANPs[1]->getValue().getMANP().getValue();

  return APIntWidthExtendUAdd(a, b);
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.neg_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::NegEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.mul_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::MulEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  mlir::Type iTy = op->getOpOperand(1).get().getType();

  assert(iTy.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(1).get().getDefiningOp());

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt sqNorm;

  if (cstOp) {
    // For a constant operand use actual constant to calculate 2-norm
    mlir::IntegerAttr attr = cstOp->getAttrOfType<mlir::IntegerAttr>("value");
    sqNorm = APIntWidthExtendSqForConstant(attr.getValue());
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUMul(sqNorm, eNorm);
}

/// Calculates the squared Minimal Arithmetic Noise Padding of an
/// `FHELinalg.add_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::AddEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::AddEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         operandMANPs[1]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt a = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt b = operandMANPs[1]->getValue().getMANP().getValue();

  return APIntWidthExtendUAdd(a, b);
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHELinalg.sub_int_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::SubIntEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[1]->getValue().getMANP().getValue();

  return eNorm;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::SubEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::SubEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  assert(operandMANPs.size() == 2 &&
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         operandMANPs[1]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt a = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt b = operandMANPs[1]->getValue().getMANP().getValue();

  return APIntWidthExtendUAdd(a, b);
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHELinalg.neg_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::NegEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.mul_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::MulEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  mlir::RankedTensorType op0Ty =
      op->getOpOperand(1).get().getType().cast<mlir::RankedTensorType>();

  mlir::Type iTy = op0Ty.getElementType();

  assert(iTy.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt sqNorm;

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(1).get().getDefiningOp());
  mlir::DenseIntElementsAttr denseVals =
      cstOp ? cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;

  if (denseVals) {
    // For a constant operand use actual constant to calculate 2-norm
    sqNorm = maxIntNorm2Sq(denseVals);
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUMul(sqNorm, eNorm);
}

static llvm::APInt computeVectorNorm(
    llvm::ArrayRef<int64_t> shape, int64_t axis,
    mlir::DenseIntElementsAttr denseValues, llvm::APInt encryptedOperandNorm,
    llvm::SmallVector<uint64_t, /*size-hint=*/4> &elementSelector) {

  llvm::APInt accumulationNorm = llvm::APInt{1, 1, false};
  for (int64_t i = 0; i < shape[axis]; i++) {
    elementSelector[axis] = i;

    auto denseValuesAP = denseValues.getValues<llvm::APInt>();
    llvm::APInt weight = denseValuesAP[elementSelector];
    llvm::APInt weightNorm = APIntWidthExtendSqForConstant(weight);

    llvm::APInt multiplicationNorm =
        APIntWidthExtendUMul(encryptedOperandNorm, weightNorm);
    accumulationNorm =
        APIntWidthExtendUAdd(multiplicationNorm, accumulationNorm);
  }
  return accumulationNorm;
}

static void determineNextVector(
    llvm::ArrayRef<int64_t> shape, int64_t destroyedDimension,
    llvm::SmallVector<uint64_t, /*size-hint=*/4> &vectorSelector) {

  for (int64_t i = shape.size() - 1; i >= 0; i--) {
    if (i == destroyedDimension) {
      continue;
    }

    if (vectorSelector[i] + 1 < (uint64_t)shape[i]) {
      vectorSelector[i]++;
      break;
    }

    vectorSelector[i] = 0;
  }
}

static llvm::APInt calculateSqManpForMatMulWithDenseValues(
    llvm::ArrayRef<int64_t> shape, int64_t destroyedDimension,
    mlir::DenseIntElementsAttr denseValues, llvm::APInt encryptedOperandNorm) {

  llvm::APInt maximumNorm = llvm::APInt{1, 1, false};

  size_t numberOfVectorsToInspect = 1;
  for (auto size : shape) {
    numberOfVectorsToInspect *= size;
  }
  numberOfVectorsToInspect /= shape[destroyedDimension];

  auto vectorSelector =
      llvm::SmallVector<uint64_t, /*size-hint=*/4>(shape.size(), 0);

  auto elementSelector = vectorSelector;
  for (size_t n = 0; n < numberOfVectorsToInspect; n++) {
    elementSelector.assign(vectorSelector);

    llvm::APInt accumulationNorm =
        computeVectorNorm(shape, destroyedDimension, denseValues,
                          encryptedOperandNorm, elementSelector);
    maximumNorm = APIntUMax(maximumNorm, accumulationNorm);

    determineNextVector(shape, destroyedDimension, vectorSelector);
  }

  return maximumNorm;
}

/// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
/// that is equivalent to an `FHE.mul_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::MatMulEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  auto lhsType =
      ((mlir::Type)op.lhs().getType()).cast<mlir::RankedTensorType>();
  auto rhsType =
      ((mlir::Type)op.rhs().getType()).cast<mlir::RankedTensorType>();

  llvm::ArrayRef<int64_t> lhsShape = lhsType.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsType.getShape();

  int64_t lhsDims = (int64_t)lhsShape.size();
  int64_t rhsDims = (int64_t)rhsShape.size();

  mlir::Type rhsElementType = rhsType.getElementType();
  assert(rhsElementType.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt lhsNorm = operandMANPs[0]->getValue().getMANP().getValue();
  llvm::APInt accNorm = llvm::APInt{1, 1, false};

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(1).get().getDefiningOp());
  mlir::DenseIntElementsAttr denseVals =
      cstOp ? cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;

  int64_t N = rhsDims <= 2 ? rhsShape[0] : rhsShape[rhsDims - 2];

  if (denseVals) {
    auto denseValsAP = denseVals.getValues<llvm::APInt>();

    if (lhsDims == 2 && rhsDims == 2) {
      // MxN @ NxP -> MxP

      int64_t M = lhsShape[0];
      int64_t P = rhsShape[1];
      for (int64_t m = 0; m < M; m++) {
        for (int64_t p = 0; p < P; p++) {
          llvm::APInt tmpNorm = llvm::APInt{1, 1, false};
          for (int64_t n = 0; n < N; n++) {
            llvm::APInt cst = denseValsAP[{(uint64_t)n, (uint64_t)p}];
            llvm::APInt rhsNorm = APIntWidthExtendSqForConstant(cst);
            llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
            tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
          }
          accNorm = APIntUMax(accNorm, tmpNorm);
        }
      }

    } else if (rhsDims == 1) {

      //     MxN @ N ->     M
      //   LxMxN @ N ->   LxM
      // KxLxMxN @ N -> KxLxM

      for (int64_t i = 0; i < N; i++) {
        llvm::APInt cst = denseValsAP[i];
        llvm::APInt rhsNorm = APIntWidthExtendSqForConstant(cst);
        llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
        accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
      }

    } else if (rhsDims >= 2) {

      // KxLxMxN @   NxP -> KxLxMxP
      // KxLxMxN @ LxNxP -> KxLxMxP
      // Kx1xMxN @ LxNxP -> KxLxMxP

      //   MxN @ KxLxNxP -> KxLxMxP
      // LxMxN @ KxLxNxP -> KxLxMxP
      // 1xMxN @ KxLxNxP -> KxLxMxP

      // N @     NxP ->     P
      // N @   LxNxP ->   LxP
      // N @ KxLxNxP -> KxLxP

      accNorm = calculateSqManpForMatMulWithDenseValues(rhsShape, rhsDims - 2,
                                                        denseVals, lhsNorm);
    }

  } else {
    llvm::APInt rhsNorm = conservativeIntNorm2Sq(rhsElementType);
    for (int64_t i = 0; i < N; i++) {
      llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
      accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
    }
  }

  return accNorm;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::MatMulIntEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  auto lhsType =
      ((mlir::Type)op.lhs().getType()).cast<mlir::RankedTensorType>();
  auto rhsType =
      ((mlir::Type)op.rhs().getType()).cast<mlir::RankedTensorType>();

  llvm::ArrayRef<int64_t> lhsShape = lhsType.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsType.getShape();

  int64_t lhsDims = (int64_t)lhsShape.size();
  int64_t rhsDims = (int64_t)rhsShape.size();

  mlir::Type lhsElementType = lhsType.getElementType();
  assert(lhsElementType.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt rhsNorm = operandMANPs[1]->getValue().getMANP().getValue();
  llvm::APInt accNorm = llvm::APInt{1, 1, false};

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(0).get().getDefiningOp());
  mlir::DenseIntElementsAttr denseVals =
      cstOp ? cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;

  int64_t N = rhsDims <= 2 ? rhsShape[0] : rhsShape[rhsDims - 2];

  if (denseVals) {
    auto denseValsAP = denseVals.getValues<llvm::APInt>();

    if (lhsDims == 2 && rhsDims == 2) {

      // MxN @ NxP -> MxP

      int64_t M = lhsShape[0];
      int64_t P = rhsShape[1];
      for (int64_t m = 0; m < M; m++) {
        for (int64_t p = 0; p < P; p++) {
          llvm::APInt tmpNorm = llvm::APInt{1, 1, false};
          for (int64_t n = 0; n < N; n++) {
            llvm::APInt cst = denseValsAP[{(uint64_t)m, (uint64_t)n}];
            llvm::APInt lhsNorm = APIntWidthExtendSqForConstant(cst);
            llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
            tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
          }
          accNorm = APIntUMax(accNorm, tmpNorm);
        }
      }

    } else if (lhsDims == 1) {

      // N @     NxP ->     P
      // N @   LxNxP ->   LxP
      // N @ KxLxNxP -> KxLxP

      for (int64_t i = 0; i < N; i++) {
        llvm::APInt cst = denseValsAP[i];
        llvm::APInt lhsNorm = APIntWidthExtendSqForConstant(cst);
        llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
        accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
      }

    } else if (lhsDims >= 2) {

      // KxLxMxN @   NxP -> KxLxMxP
      // KxLxMxN @ LxNxP -> KxLxMxP
      // Kx1xMxN @ LxNxP -> KxLxMxP

      //   MxN @ KxLxNxP -> KxLxMxP
      // LxMxN @ KxLxNxP -> KxLxMxP
      // 1xMxN @ KxLxNxP -> KxLxMxP

      //     MxN @ N ->     M
      //   LxMxN @ N ->   LxM
      // KxLxMxN @ N -> KxLxM

      accNorm = calculateSqManpForMatMulWithDenseValues(lhsShape, lhsDims - 1,
                                                        denseVals, rhsNorm);
    }

  } else {
    llvm::APInt lhsNorm = conservativeIntNorm2Sq(lhsElementType);
    for (int64_t i = 0; i < N; i++) {
      llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
      accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
    }
  }

  return accNorm;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::TransposeOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() == 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().getValue();
}

static llvm::APInt getSqMANP(
    mlir::tensor::ExtractOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[0]->getValue().getMANP().getValue();

  return eNorm;
}

static llvm::APInt getSqMANP(
    FHELinalg::FromElementOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  auto manp = operandMANPs[0]->getValue().getMANP();
  if (manp.hasValue()) {
    return manp.getValue();
  }

  return llvm::APInt{1, 1, false};
}

static llvm::APInt getSqMANP(
    mlir::tensor::FromElementsOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  auto max = std::max_element(
      operandMANPs.begin(), operandMANPs.end(),
      [](mlir::LatticeElement<MANPLatticeValue> *const a,
         mlir::LatticeElement<MANPLatticeValue> *const b) {
        return APIntWidthExtendULT(a->getValue().getMANP().getValue(),
                                   b->getValue().getMANP().getValue());
      });
  return (*max)->getValue().getMANP().getValue();
}

static llvm::APInt getSqMANP(
    mlir::tensor::ExtractSliceOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().getValue();
}

static llvm::APInt getSqMANP(
    mlir::tensor::InsertSliceOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() >= 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return APIntUMax(operandMANPs[0]->getValue().getMANP().getValue(),
                   operandMANPs[1]->getValue().getMANP().getValue());
}

static llvm::APInt getSqMANP(
    mlir::tensor::InsertOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() >= 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return APIntUMax(operandMANPs[0]->getValue().getMANP().getValue(),
                   operandMANPs[1]->getValue().getMANP().getValue());
}

static llvm::APInt getSqMANP(
    mlir::tensor::CollapseShapeOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() >= 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().getValue();
}

static llvm::APInt getSqMANP(
    mlir::tensor::ExpandShapeOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() >= 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().getValue();
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::SumOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  auto inputType = op.getOperand().getType().dyn_cast<mlir::TensorType>();

  uint64_t numberOfElementsInTheInput = inputType.getNumElements();
  if (numberOfElementsInTheInput == 0) {
    return llvm::APInt{1, 1, false};
  }

  uint64_t numberOfElementsAddedTogetherInEachOutputCell = 1;

  mlir::ArrayAttr axes = op.axes();
  if (axes.empty()) {
    numberOfElementsAddedTogetherInEachOutputCell *= numberOfElementsInTheInput;
  } else {
    llvm::ArrayRef<int64_t> shape = inputType.getShape();
    for (mlir::Attribute axisAttribute : op.axes()) {
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
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operands");

  llvm::APInt operandMANP = operandMANPs[0]->getValue().getMANP().getValue();

  return APIntWidthExtendUMul(noiseMultiplier, operandMANP);
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::ConcatOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  llvm::APInt result = llvm::APInt{1, 0, false};
  for (mlir::LatticeElement<MANPLatticeValue> *operandMANP : operandMANPs) {
    llvm::APInt candidate = operandMANP->getValue().getMANP().getValue();
    if (candidate.getLimitedValue() >= result.getLimitedValue()) {
      result = candidate;
    }
  }
  return result;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::Conv2dOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  mlir::RankedTensorType weightTy =
      op.weight().getType().cast<mlir::RankedTensorType>();

  mlir::Type weightIntType = weightTy.getElementType();

  // Bias is optional, so we can have both 2 or 3 operands
  assert((operandMANPs.size() == 2 || operandMANPs.size() == 3) &&
         operandMANPs[0]->getValue().getMANP().hasValue() &&
         "Missing squared Minimal Arithmetic Noise Padding for encrypted "
         "operand");

  llvm::APInt inputNorm = operandMANPs[0]->getValue().getMANP().getValue();

  mlir::arith::ConstantOp weightCstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(1).get().getDefiningOp());
  mlir::DenseIntElementsAttr weightDenseVals =
      weightCstOp
          ? weightCstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
          : nullptr;

  mlir::DenseIntElementsAttr biasDenseVals = nullptr;
  mlir::Type biasIntType;
  bool hasBias = operandMANPs.size() == 3;
  if (hasBias) {
    biasIntType =
        op.bias().getType().cast<mlir::RankedTensorType>().getElementType();
    mlir::arith::ConstantOp biasCstOp =
        llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
            op->getOpOperand(2).get().getDefiningOp());
    biasDenseVals =
        biasCstOp
            ? biasCstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;
  }

  // Initial value of the accumulator to 0, or the conservative norm of the bias
  // if there is a non-const bias
  llvm::APInt accNorm;
  if (hasBias && biasDenseVals == nullptr) {
    accNorm = conservativeIntNorm2Sq(biasIntType);
  } else {
    accNorm = llvm::APInt{0, 1, false};
  }

  // Weight shapes: Filter*Channel*Height*Width
  uint64_t F = weightTy.getShape()[0];
  uint64_t C = weightTy.getShape()[1];
  uint64_t H = weightTy.getShape()[2];
  uint64_t W = weightTy.getShape()[3];
  if (weightDenseVals) {
    auto weightDenseValsAP = weightDenseVals.getValues<llvm::APInt>();
    // For a constant weight kernel use actual constant to calculate 2-norm
    // input windows are being multiplied by a kernel and summed up
    for (uint64_t f = 0; f < F; f++) {
      llvm::APInt tmpNorm = accNorm;
      // If there is a bias, start accumulating from its norm
      if (hasBias && biasDenseVals) {
        llvm::APInt cst = biasDenseVals.getValues<llvm::APInt>()[f];
        tmpNorm = APIntWidthExtendSqForConstant(cst);
      }
      for (uint64_t c = 0; c < C; c++) {
        for (uint64_t h = 0; h < H; h++) {
          for (uint64_t w = 0; w < W; w++) {
            llvm::APInt cst = weightDenseValsAP[{f, c, h, w}];
            llvm::APInt weightNorm = APIntWidthExtendSqForConstant(cst);
            llvm::APInt mulNorm = APIntWidthExtendUMul(inputNorm, weightNorm);
            tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
          }
        }
      }
      accNorm = APIntUMax(accNorm, tmpNorm);
    }
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    llvm::APInt weightNorm = conservativeIntNorm2Sq(weightIntType);
    // For a weight (kernel) of shape tensor<FxCxHxW>, there is C*H*W
    // FHE.mul_eint_int and FHE.add_eint operations for each elements of the
    // result
    int64_t n_mul = C * H * W;
    llvm::APInt tmpNorm = llvm::APInt{1, 1, false};
    for (int64_t i = 0; i < n_mul; i++) {
      llvm::APInt mulNorm = APIntWidthExtendUMul(inputNorm, weightNorm);
      tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
    }
    if (hasBias && biasDenseVals) {
      auto biasDenseValsAP = biasDenseVals.getValues<llvm::APInt>();
      llvm::APInt maxNorm = tmpNorm;
      for (uint64_t f = 0; f < F; f++) {
        llvm::APInt cst = biasDenseValsAP[f];
        llvm::APInt currentNorm = APIntWidthExtendSqForConstant(cst);
        currentNorm = APIntWidthExtendUAdd(currentNorm, tmpNorm);
        maxNorm = APIntUMax(currentNorm, maxNorm);
      }
      tmpNorm = maxNorm;
    }
    accNorm = APIntWidthExtendUAdd(accNorm, tmpNorm);
  }
  return accNorm;
}

struct MANPAnalysis : public mlir::ForwardDataFlowAnalysis<MANPLatticeValue> {
  using ForwardDataFlowAnalysis<MANPLatticeValue>::ForwardDataFlowAnalysis;
  MANPAnalysis(mlir::MLIRContext *ctx, bool debug)
      : mlir::ForwardDataFlowAnalysis<MANPLatticeValue>(ctx), debug(debug) {}

  ~MANPAnalysis() override = default;

  mlir::ChangeResult visitOperation(
      mlir::Operation *op,
      llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operands) final {
    mlir::LatticeElement<MANPLatticeValue> &latticeRes =
        getLatticeElement(op->getResult(0));
    bool isDummy = false;
    llvm::APInt norm2SqEquiv;

    // FHE Operators
    if (auto addEintIntOp =
            llvm::dyn_cast<mlir::concretelang::FHE::AddEintIntOp>(op)) {
      norm2SqEquiv = getSqMANP(addEintIntOp, operands);
    } else if (auto addEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::AddEintOp>(op)) {
      norm2SqEquiv = getSqMANP(addEintOp, operands);
    } else if (auto subIntEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::SubIntEintOp>(op)) {
      norm2SqEquiv = getSqMANP(subIntEintOp, operands);
    } else if (auto subEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::SubEintIntOp>(op)) {
      norm2SqEquiv = getSqMANP(subEintIntOp, operands);
    } else if (auto subEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::SubEintOp>(op)) {
      norm2SqEquiv = getSqMANP(subEintOp, operands);
    } else if (auto negEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::NegEintOp>(op)) {
      norm2SqEquiv = getSqMANP(negEintOp, operands);
    } else if (auto mulEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::MulEintIntOp>(op)) {
      norm2SqEquiv = getSqMANP(mulEintIntOp, operands);
    } else if (llvm::isa<mlir::concretelang::FHE::ZeroEintOp>(op) ||
               llvm::isa<mlir::concretelang::FHE::ZeroTensorOp>(op) ||
               llvm::isa<mlir::concretelang::FHE::ApplyLookupTableEintOp>(op)) {
      norm2SqEquiv = llvm::APInt{1, 1, false};
    }
    // FHELinalg Operators
    else if (auto dotOp =
                 llvm::dyn_cast<mlir::concretelang::FHELinalg::Dot>(op)) {
      norm2SqEquiv = getSqMANP(dotOp, operands);
    } else if (auto addEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::AddEintIntOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(addEintIntOp, operands);
    } else if (auto addEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::AddEintOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(addEintOp, operands);
    } else if (auto subIntEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::SubIntEintOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(subIntEintOp, operands);
    } else if (auto subEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::SubEintIntOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(subEintIntOp, operands);
    } else if (auto subEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::SubEintOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(subEintOp, operands);
    } else if (auto negEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::NegEintOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(negEintOp, operands);
    } else if (auto mulEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::MulEintIntOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(mulEintIntOp, operands);
    } else if (auto matmulEintIntOp = llvm::dyn_cast<
                   mlir::concretelang::FHELinalg::MatMulEintIntOp>(op)) {
      norm2SqEquiv = getSqMANP(matmulEintIntOp, operands);
    } else if (auto matmulIntEintOp = llvm::dyn_cast<
                   mlir::concretelang::FHELinalg::MatMulIntEintOp>(op)) {
      norm2SqEquiv = getSqMANP(matmulIntEintOp, operands);
    } else if (llvm::isa<
                   mlir::concretelang::FHELinalg::ApplyLookupTableEintOp,
                   mlir::concretelang::FHELinalg::ApplyMultiLookupTableEintOp,
                   mlir::concretelang::FHELinalg::ApplyMappedLookupTableEintOp>(
                   op)) {
      norm2SqEquiv = llvm::APInt{1, 1, false};
    } else if (auto sumOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::SumOp>(op)) {
      norm2SqEquiv = getSqMANP(sumOp, operands);
    } else if (auto concatOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::ConcatOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(concatOp, operands);
    } else if (auto conv2dOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::Conv2dOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(conv2dOp, operands);
    } else if (auto fromElementOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::FromElementOp>(
                       op)) {
      norm2SqEquiv = getSqMANP(fromElementOp, operands);
    } else if (auto transposeOp =
                   llvm::dyn_cast<mlir::concretelang::FHELinalg::TransposeOp>(
                       op)) {
      if (transposeOp.tensor()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(transposeOp, operands);
      } else {
        isDummy = true;
      }
    }

    // Tensor Operators
    // ExtractOp
    else if (auto extractOp = llvm::dyn_cast<mlir::tensor::ExtractOp>(op)) {
      if (extractOp.result()
              .getType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(extractOp, operands);
      } else {
        isDummy = true;
      }
    }
    // ExtractSliceOp
    else if (auto extractSliceOp =
                 llvm::dyn_cast<mlir::tensor::ExtractSliceOp>(op)) {
      if (extractSliceOp.result()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(extractSliceOp, operands);
      } else {
        isDummy = true;
      }
    }
    // InsertOp
    else if (auto insertOp = llvm::dyn_cast<mlir::tensor::InsertOp>(op)) {
      if (insertOp.result()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(insertOp, operands);
      } else {
        isDummy = true;
      }
    }
    // InsertSliceOp
    else if (auto insertSliceOp =
                 llvm::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
      if (insertSliceOp.result()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(insertSliceOp, operands);
      } else {
        isDummy = true;
      }
    }
    // FromElementOp
    else if (auto fromOp = llvm::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
      if (fromOp.result()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(fromOp, operands);
      } else {
        isDummy = true;
      }
    }
    // TensorCollapseShapeOp
    else if (auto reshapeOp =
                 llvm::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
      if (reshapeOp.result()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(reshapeOp, operands);
      } else {
        isDummy = true;
      }
    }
    // TensorExpandShapeOp
    else if (auto reshapeOp = llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
      if (reshapeOp.result()
              .getType()
              .cast<mlir::TensorType>()
              .getElementType()
              .isa<mlir::concretelang::FHE::EncryptedIntegerType>()) {
        norm2SqEquiv = getSqMANP(reshapeOp, operands);
      } else {
        isDummy = true;
      }
    }

    else if (llvm::isa<mlir::arith::ConstantOp>(op)) {
      isDummy = true;
    } else if (llvm::isa<mlir::concretelang::FHE::FHEDialect>(
                   *op->getDialect())) {
      op->emitError("Unsupported operation");
      assert(false && "Unsupported operation");
    } else {
      isDummy = true;
    }

    if (!isDummy) {
      latticeRes.join(MANPLatticeValue{norm2SqEquiv});
      latticeRes.markOptimisticFixpoint();

      op->setAttr("SMANP",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(
                          op->getContext(), norm2SqEquiv.getBitWidth(),
                          mlir::IntegerType::SignednessSemantics::Unsigned),
                      norm2SqEquiv));

      llvm::APInt norm2Equiv = APIntCeilSqrt(norm2SqEquiv);

      op->setAttr("MANP",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(
                          op->getContext(), norm2Equiv.getBitWidth(),
                          mlir::IntegerType::SignednessSemantics::Unsigned),
                      norm2Equiv));

      if (debug) {
        op->emitRemark("Squared Minimal Arithmetic Noise Padding: ")
            << APIntToStringValUnsigned(norm2SqEquiv) << "\n";
      }
    } else {
      latticeRes.join(MANPLatticeValue{});
    }

    return mlir::ChangeResult::Change;
  }

private:
  bool debug;
};
} // namespace

namespace {
/// For documentation see MANP.td
struct MANPPass : public MANPBase<MANPPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    MANPAnalysis analysis(func->getContext(), debug);
    analysis.run(func);
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
  MaxMANPPass(std::function<void(const llvm::APInt &, unsigned)> updateMax)
      : updateMax(updateMax), maxMANP(llvm::APInt{1, 0, false}),
        maxEintWidth(0){};

protected:
  void processOperation(mlir::Operation *op) {
    static const llvm::APInt one{1, 1, false};
    bool upd = false;

    // Process all function arguments and use the default value of 1
    // for MANP and the declarend precision
    if (mlir::func::FuncOp func =
            llvm::dyn_cast_or_null<mlir::func::FuncOp>(op)) {
      for (mlir::BlockArgument blockArg : func.getBody().getArguments()) {
        if (isEncryptedFunctionParameter(blockArg)) {
          unsigned int width = fhe::utils::getEintPrecision(blockArg);

          if (this->maxEintWidth < width) {
            this->maxEintWidth = width;
          }

          if (APIntWidthExtendULT(this->maxMANP, one)) {
            this->maxMANP = one;
            upd = true;
          }
        }
      }
    }

    // Process all results using MANP attribute from MANP pas
    for (mlir::OpResult res : op->getResults()) {
      mlir::concretelang::FHE::EncryptedIntegerType eTy =
          res.getType()
              .dyn_cast_or_null<
                  mlir::concretelang::FHE::EncryptedIntegerType>();
      if (eTy == nullptr) {
        auto tensorTy = res.getType().dyn_cast_or_null<mlir::TensorType>();
        if (tensorTy != nullptr) {
          eTy = tensorTy.getElementType()
                    .dyn_cast_or_null<
                        mlir::concretelang::FHE::EncryptedIntegerType>();
        }
      }

      if (eTy) {
        if (this->maxEintWidth < eTy.getWidth()) {
          this->maxEintWidth = eTy.getWidth();
          upd = true;
        }

        mlir::IntegerAttr MANP = op->getAttrOfType<mlir::IntegerAttr>("MANP");

        if (!MANP) {
          op->emitError("Maximum Arithmetic Noise Padding value not set");
          this->signalPassFailure();
          return;
        }

        if (APIntWidthExtendULT(this->maxMANP, MANP.getValue())) {
          this->maxMANP = MANP.getValue();
          upd = true;
        }
      }
    }

    if (upd)
      this->updateMax(this->maxMANP, this->maxEintWidth);
  }

  std::function<void(const llvm::APInt &, unsigned)> updateMax;
  llvm::APInt maxMANP;
  unsigned int maxEintWidth;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createMaxMANPPass(
    std::function<void(const llvm::APInt &, unsigned)> updateMax) {
  return std::make_unique<MaxMANPPass>(updateMax);
}

} // namespace concretelang
} // namespace mlir
