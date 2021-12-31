// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include <concretelang/Dialect/FHE/Analysis/MANP.h>
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
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
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

// Returns `true` if the given value is a scalar or tensor argument of
// a function, for which a MANP of 1 can be assumed.
static bool isEncryptedFunctionParameter(mlir::Value value) {
  if (!value.isa<mlir::BlockArgument>())
    return false;

  mlir::Block *block = value.cast<mlir::BlockArgument>().getOwner();

  if (!block || !block->getParentOp() ||
      !llvm::isa<mlir::FuncOp>(block->getParentOp())) {
    return false;
  }

  return (
      value.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>() ||
      (value.getType().isa<mlir::TensorType>() &&
       value.getType()
           .cast<mlir::TensorType>()
           .getElementType()
           .isa<mlir::concretelang::FHE::EncryptedIntegerType>()));
}

// Returns the bit width of `value` if `value` is an encrypted integer
// or the bit width of the elements if `value` is a tensor of
// encrypted integers.
static unsigned int getEintPrecision(mlir::Value value) {
  if (auto ty = value.getType()
                    .dyn_cast_or_null<
                        mlir::concretelang::FHE::EncryptedIntegerType>()) {
    return ty.getWidth();
  } else if (auto tensorTy =
                 value.getType().dyn_cast_or_null<mlir::TensorType>()) {
    if (auto ty = tensorTy.getElementType()
                      .dyn_cast_or_null<
                          mlir::concretelang::FHE::EncryptedIntegerType>())
      return ty.getWidth();
  }

  assert(false &&
         "Value is neither an encrypted integer nor a tensor of encrypted "
         "integers");

  return 0;
}

// The `MANPLatticeValue` represents the squared Minimal Arithmetic
// Noise Padding for an operation using the squared 2-norm of an
// equivalent dot operation. This can either be an actual value if the
// values for its predecessors have been calculated beforehand or an
// unknown value otherwise.
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

  // Required by `mlir::LatticeElement::join()`, but should never be
  // invoked, as `MANPAnalysis::visitOperation()` takes care of
  // combining the squared Minimal Arithmetic Noise Padding of
  // operands into the Minimal Arithmetic Noise Padding of the result.
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

// Checks if `lhs` is less than `rhs`, where both values are assumed
// to be positive. The bit width of the smaller `APInt` is extended
// before comparison via `APInt::ult`.
static bool APIntWidthExtendULT(const llvm::APInt &lhs,
                                const llvm::APInt &rhs) {
  if (lhs.getBitWidth() < rhs.getBitWidth())
    return lhs.zext(rhs.getBitWidth()).ult(rhs);
  else if (lhs.getBitWidth() > rhs.getBitWidth())
    return lhs.ult(rhs.zext(lhs.getBitWidth()));
  else
    return lhs.ult(rhs);
}

// Adds two `APInt` values, where both values are assumed to be
// positive. The bit width of the operands is extended in order to
// guarantee that the sum fits into the resulting `APInt`.
static llvm::APInt APIntWidthExtendUAdd(const llvm::APInt &lhs,
                                        const llvm::APInt &rhs) {
  unsigned maxBits = std::max(lhs.getBitWidth(), rhs.getBitWidth());

  // Make sure the required number of bits can be represented by the
  // `unsigned` argument of `zext`.
  assert(std::numeric_limits<unsigned>::max() - maxBits > 1);

  unsigned targetWidth = maxBits + 1;
  return lhs.zext(targetWidth) + rhs.zext(targetWidth);
}

// Multiplies two `APInt` values, where both values are assumed to be
// positive. The bit width of the operands is extended in order to
// guarantee that the product fits into the resulting `APInt`.
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

// Returns the maximum value beetwen `lhs` and `rhs`, where both values are
// assumed to be positive. The bit width of the smaller `APInt` is extended
// before comparison via `APInt::ult`.
static llvm::APInt APIntUMax(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  if (APIntWidthExtendULT(lhs, rhs)) {
    return rhs;
  }
  return lhs;
}

// Calculates the square of `i`. The bit width `i` is extended in
// order to guarantee that the product fits into the resulting
// `APInt`.
static llvm::APInt APIntWidthExtendUSq(const llvm::APInt &i) {
  // Make sure the required number of bits can be represented by the
  // `unsigned` argument of `zext`.
  assert(i.getBitWidth() < std::numeric_limits<unsigned>::max() / 2 &&
         "Required number of bits cannot be represented with an APInt");

  llvm::APInt ie = i.zext(2 * i.getBitWidth());

  return ie * ie;
}

// Calculates the square root of `i` and rounds it to the next highest
// integer value (i.e., the square of the result is guaranteed to be
// greater or equal to `i`).
static llvm::APInt APIntCeilSqrt(const llvm::APInt &i) {
  llvm::APInt res = i.sqrt();
  llvm::APInt resSq = APIntWidthExtendUSq(res);

  if (APIntWidthExtendULT(resSq, i))
    return APIntWidthExtendUAdd(res, llvm::APInt{1, 1, false});
  else
    return res;
}

// Returns a string representation of `i` assuming that `i` is an
// unsigned value.
static std::string APIntToStringValUnsigned(const llvm::APInt &i) {
  llvm::SmallString<32> s;
  i.toStringUnsigned(s);
  return std::string(s.c_str());
}

// Calculates the square of the 2-norm of a tensor initialized with a
// dense matrix of constant, signless integers. Aborts if the value
// type or initialization of of `cstOp` is incorrect.
static llvm::APInt denseCstTensorNorm2Sq(mlir::arith::ConstantOp cstOp,
                                         llvm::APInt eNorm) {
  mlir::DenseIntElementsAttr denseVals =
      cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value");

  assert(denseVals && cstOp.getType().isa<mlir::TensorType>() &&
         "Constant must be a tensor initialized with `dense`");

  mlir::TensorType tensorType = cstOp.getType().cast<mlir::TensorType>();

  assert(tensorType.getElementType().isSignlessInteger() &&
         "Can only handle tensors with signless integer elements");

  mlir::IntegerType elementType =
      tensorType.getElementType().cast<mlir::IntegerType>();

  llvm::APInt accu{1, 0, false};

  for (llvm::APInt val : denseVals.getValues<llvm::APInt>()) {
    llvm::APInt valSqNorm = APIntWidthExtendUSq(val);
    llvm::APInt mulSqNorm = APIntWidthExtendUMul(valSqNorm, eNorm);
    accu = APIntWidthExtendUAdd(accu, mulSqNorm);
  }

  return accu;
}

// Calculates the square of the 2-norm of a 1D tensor of signless
// integers by conservatively assuming that the dynamic values are the
// maximum for the integer width. Aborts if the tensor type `tTy` is
// incorrect.
static llvm::APInt denseDynTensorNorm2Sq(mlir::TensorType tTy,
                                         llvm::APInt eNorm) {
  assert(tTy && tTy.getElementType().isSignlessInteger() &&
         tTy.hasStaticShape() && tTy.getRank() == 1 &&
         "Plaintext operand must be a statically shaped 1D tensor of integers");

  // Make sure the log2 of the number of elements fits into an
  // unsigned
  assert(std::numeric_limits<unsigned>::max() > 8 * sizeof(uint64_t));

  unsigned elWidth = tTy.getElementTypeBitWidth();

  llvm::APInt maxVal = APInt::getMaxValue(elWidth);
  llvm::APInt maxValSq = APIntWidthExtendUSq(maxVal);
  llvm::APInt maxMulSqNorm = APIntWidthExtendUMul(maxValSq, eNorm);

  // Calculate number of bits for APInt to store number of elements
  uint64_t nElts = (uint64_t)tTy.getNumElements();
  assert(std::numeric_limits<int64_t>::max() - nElts > 1);
  unsigned nEltsBits = (unsigned)ceilLog2(nElts + 1);

  llvm::APInt nEltsAP{nEltsBits, nElts, false};

  return APIntWidthExtendUMul(maxMulSqNorm, nEltsAP);
}

// Calculates the squared Minimal Arithmetic Noise Padding of an
// `FHELinalg.dot_eint_int` operation.
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

// Returns the squared 2-norm for a dynamic integer by conservatively
// assuming that the integer's value is the maximum for the integer
// width.
static llvm::APInt conservativeIntNorm2Sq(mlir::Type t) {
  assert(t.isSignlessInteger() && "Type must be a signless integer type");
  assert(std::numeric_limits<unsigned>::max() - t.getIntOrFloatBitWidth() > 1);

  llvm::APInt maxVal{t.getIntOrFloatBitWidth() + 1, 1, false};
  maxVal <<= t.getIntOrFloatBitWidth();
  return APIntWidthExtendUSq(maxVal);
}

// Calculates the squared Minimal Arithmetic Noise Padding of an
// `FHE.add_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::AddEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  mlir::Type iTy = op->getOpOperand(1).get().getType();

  assert(iTy.isSignlessInteger() &&
         "Only additions with signless integers are currently allowed");

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
    sqNorm = APIntWidthExtendUSq(attr.getValue());
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUAdd(sqNorm, eNorm);
}

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHE.add_eint` operation.
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

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHE.sub_int_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHE::SubIntEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {
  mlir::Type iTy = op->getOpOperand(0).get().getType();

  assert(iTy.isSignlessInteger() &&
         "Only subtractions with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[1]->getValue().getMANP().getValue();
  llvm::APInt sqNorm;

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(0).get().getDefiningOp());

  if (cstOp) {
    // For constant plaintext operands simply use the constant value
    mlir::IntegerAttr attr = cstOp->getAttrOfType<mlir::IntegerAttr>("value");
    sqNorm = APIntWidthExtendUSq(attr.getValue());
  } else {
    // For dynamic plaintext operands conservatively assume that the integer has
    // its maximum possible value
    sqNorm = conservativeIntNorm2Sq(iTy);
  }
  return APIntWidthExtendUAdd(sqNorm, eNorm);
}

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHE.neg_eint` operation.
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

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHE.mul_eint_int` operation.
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
    sqNorm = APIntWidthExtendUSq(attr.getValue());
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUMul(sqNorm, eNorm);
}

// Calculates the squared Minimal Arithmetic Noise Padding of an
// `FHELinalg.add_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::AddEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  mlir::RankedTensorType op1Ty =
      op->getOpOperand(1).get().getType().cast<mlir::RankedTensorType>();

  mlir::Type iTy = op1Ty.getElementType();

  assert(iTy.isSignlessInteger() &&
         "Only additions with signless integers are currently allowed");

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
    llvm::APInt maxCst = denseVals.getFlatValue<llvm::APInt>(0);
    for (int64_t i = 0; i < denseVals.getNumElements(); i++) {
      llvm::APInt iCst = denseVals.getFlatValue<llvm::APInt>(i);
      if (maxCst.ult(iCst)) {
        maxCst = iCst;
      }
    }
    sqNorm = APIntWidthExtendUSq(maxCst);
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUAdd(sqNorm, eNorm);
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

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHELinalg.sub_int_eint` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::SubIntEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  mlir::RankedTensorType op0Ty =
      op->getOpOperand(0).get().getType().cast<mlir::RankedTensorType>();

  mlir::Type iTy = op0Ty.getElementType();

  assert(iTy.isSignlessInteger() &&
         "Only subtractions with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt eNorm = operandMANPs[1]->getValue().getMANP().getValue();
  llvm::APInt sqNorm;

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(0).get().getDefiningOp());
  mlir::DenseIntElementsAttr denseVals =
      cstOp ? cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;

  if (denseVals) {
    // For a constant operand use actual constant to calculate 2-norm
    llvm::APInt maxCst = denseVals.getFlatValue<llvm::APInt>(0);
    for (int64_t i = 0; i < denseVals.getNumElements(); i++) {
      llvm::APInt iCst = denseVals.getFlatValue<llvm::APInt>(i);
      if (maxCst.ult(iCst)) {
        maxCst = iCst;
      }
    }
    sqNorm = APIntWidthExtendUSq(maxCst);
  } else {
    // For dynamic plaintext operands conservatively assume that the integer has
    // its maximum possible value
    sqNorm = conservativeIntNorm2Sq(iTy);
  }
  return APIntWidthExtendUAdd(sqNorm, eNorm);
}

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHELinalg.neg_eint` operation.
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

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHE.mul_eint_int` operation.
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
    llvm::APInt maxCst = denseVals.getFlatValue<llvm::APInt>(0);
    for (int64_t i = 0; i < denseVals.getNumElements(); i++) {
      llvm::APInt iCst = denseVals.getFlatValue<llvm::APInt>(i);
      if (maxCst.ult(iCst)) {
        maxCst = iCst;
      }
    }
    sqNorm = APIntWidthExtendUSq(maxCst);
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    sqNorm = conservativeIntNorm2Sq(iTy);
  }

  return APIntWidthExtendUMul(sqNorm, eNorm);
}

// Calculates the squared Minimal Arithmetic Noise Padding of a dot operation
// that is equivalent to an `FHE.mul_eint_int` operation.
static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::MatMulEintIntOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  mlir::RankedTensorType rhsTy =
      op.rhs().getType().cast<mlir::RankedTensorType>();
  mlir::RankedTensorType lhsTy =
      op.lhs().getType().cast<mlir::RankedTensorType>();

  mlir::Type iTy = rhsTy.getElementType();

  assert(iTy.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt lhsNorm = operandMANPs[0]->getValue().getMANP().getValue();
  // Initial value of the accumulator
  llvm::APInt accNorm = llvm::APInt{1, 1, false};

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(1).get().getDefiningOp());
  mlir::DenseIntElementsAttr denseVals =
      cstOp ? cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;

  if (denseVals) {
    // For a constant operand use actual constant to calculate 2-norm
    // tensor<MxN> = tensor<MxP> * tensor<PxN> compute the max 2-norm of the
    // result
    int64_t M = lhsTy.getShape()[0];
    int64_t N = rhsTy.getShape()[1];
    int64_t P = rhsTy.getShape()[0];
    for (int64_t m = 0; m < M; m++) {
      for (int64_t n = 0; n < N; n++) {
        llvm::APInt tmpNorm = llvm::APInt{1, 1, false};
        for (int64_t p = 0; p < P; p++) {
          llvm::APInt cst = denseVals.getFlatValue<llvm::APInt>(p * N + n);
          llvm::APInt rhsNorm = APIntWidthExtendUSq(cst);
          llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
          tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
        }
        accNorm = APIntUMax(accNorm, tmpNorm);
      }
    }
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    llvm::APInt rhsNorm = conservativeIntNorm2Sq(iTy);
    // For tensor<MxN> = tensor<MxP> * tensor<PxN> they are P FHE.mul_eint_int
    // and FHE.add_eint operations for each elements of the result
    int64_t P = rhsTy.getShape()[0];
    for (int64_t i = 0; i < P; i++) {
      llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
      accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
    }
  }

  return accNorm;
}

static llvm::APInt getSqMANP(
    mlir::concretelang::FHELinalg::MatMulIntEintOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  mlir::RankedTensorType rhsTy =
      op.rhs().getType().cast<mlir::RankedTensorType>();
  mlir::RankedTensorType lhsTy =
      op.lhs().getType().cast<mlir::RankedTensorType>();

  mlir::Type iTy = lhsTy.getElementType();

  assert(iTy.isSignlessInteger() &&
         "Only multiplications with signless integers are currently allowed");

  assert(
      operandMANPs.size() == 2 &&
      operandMANPs[1]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  llvm::APInt rhsNorm = operandMANPs[1]->getValue().getMANP().getValue();
  // Initial value of the accumulator
  llvm::APInt accNorm = llvm::APInt{1, 1, false};

  mlir::arith::ConstantOp cstOp =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op->getOpOperand(0).get().getDefiningOp());
  mlir::DenseIntElementsAttr denseVals =
      cstOp ? cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value")
            : nullptr;

  if (denseVals) {
    // For a constant operand use actual constant to calculate 2-norm
    // tensor<MxN> = tensor<MxP> * tensor<PxN> compute the max 2-norm of the
    // result
    int64_t M = lhsTy.getShape()[0];
    int64_t N = rhsTy.getShape()[1];
    int64_t P = rhsTy.getShape()[0];
    for (int64_t m = 0; m < M; m++) {
      for (int64_t n = 0; n < N; n++) {
        llvm::APInt tmpNorm = llvm::APInt{1, 1, false};
        for (int64_t p = 0; p < P; p++) {
          llvm::APInt cst = denseVals.getFlatValue<llvm::APInt>(m * P + p);
          llvm::APInt lhsNorm = APIntWidthExtendUSq(cst);
          llvm::APInt mulNorm = APIntWidthExtendUMul(lhsNorm, rhsNorm);
          tmpNorm = APIntWidthExtendUAdd(mulNorm, tmpNorm);
        }
        accNorm = APIntUMax(accNorm, tmpNorm);
      }
    }
  } else {
    // For a dynamic operand conservatively assume that the value is
    // the maximum for the integer width
    llvm::APInt lhsNorm = conservativeIntNorm2Sq(iTy);
    // For tensor<MxN> = tensor<MxP> * tensor<PxN> they are P FHE.mul_eint_int
    // and FHE.add_eint operations for each elements of the result
    int64_t P = rhsTy.getShape()[0];
    for (int64_t i = 0; i < P; i++) {
      llvm::APInt mulNorm = APIntWidthExtendUMul(rhsNorm, lhsNorm);
      accNorm = APIntWidthExtendUAdd(mulNorm, accNorm);
    }
  }

  return accNorm;
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
    mlir::linalg::TensorCollapseShapeOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() >= 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().getValue();
}

static llvm::APInt getSqMANP(
    mlir::linalg::TensorExpandShapeOp op,
    llvm::ArrayRef<mlir::LatticeElement<MANPLatticeValue> *> operandMANPs) {

  assert(
      operandMANPs.size() >= 1 &&
      operandMANPs[0]->getValue().getMANP().hasValue() &&
      "Missing squared Minimal Arithmetic Noise Padding for encrypted operand");

  return operandMANPs[0]->getValue().getMANP().getValue();
}

struct MANPAnalysis : public mlir::ForwardDataFlowAnalysis<MANPLatticeValue> {
  using ForwardDataFlowAnalysis<MANPLatticeValue>::ForwardDataFlowAnalysis;
  MANPAnalysis(mlir::MLIRContext *ctx, bool debug)
      : debug(debug), mlir::ForwardDataFlowAnalysis<MANPLatticeValue>(ctx) {}

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
    } else if (auto negEintOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::NegEintOp>(op)) {
      norm2SqEquiv = getSqMANP(negEintOp, operands);
    } else if (auto mulEintIntOp =
                   llvm::dyn_cast<mlir::concretelang::FHE::MulEintIntOp>(op)) {
      norm2SqEquiv = getSqMANP(mulEintIntOp, operands);
    } else if (llvm::isa<mlir::concretelang::FHE::ZeroEintOp>(op) ||
               llvm::isa<mlir::concretelang::FHELinalg::ZeroOp>(op) ||
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
                 llvm::dyn_cast<mlir::linalg::TensorCollapseShapeOp>(op)) {
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
    else if (auto reshapeOp =
                 llvm::dyn_cast<mlir::linalg::TensorExpandShapeOp>(op)) {
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
// For documentation see MANP.td
struct MANPPass : public MANPBase<MANPPass> {
  void runOnFunction() override {
    mlir::FuncOp func = getFunction();

    MANPAnalysis analysis(func->getContext(), debug);
    analysis.run(func);
  }
  MANPPass() = delete;
  MANPPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

// Create an instance of the Minimal Arithmetic Noise Padding analysis
// pass. If `debug` is true, for each operation, the pass emits a
// remark containing the squared Minimal Arithmetic Noise Padding of
// the equivalent dot operation.
std::unique_ptr<mlir::Pass> createMANPPass(bool debug) {
  return std::make_unique<MANPPass>(debug);
}

namespace {
// For documentation see MANP.td
struct MaxMANPPass : public MaxMANPBase<MaxMANPPass> {
  void runOnFunction() override {
    mlir::FuncOp func = getFunction();

    func.walk(
        [&](mlir::Operation *childOp) { this->processOperation(childOp); });
  }
  MaxMANPPass() = delete;
  MaxMANPPass(std::function<void(const llvm::APInt &, unsigned)> updateMax)
      : maxMANP(llvm::APInt{1, 0, false}), maxEintWidth(0),
        updateMax(updateMax){};

protected:
  void processOperation(mlir::Operation *op) {
    static const llvm::APInt one{1, 1, false};
    bool upd = false;

    // Process all function arguments and use the default value of 1
    // for MANP and the declarend precision
    if (mlir::FuncOp func = llvm::dyn_cast_or_null<mlir::FuncOp>(op)) {
      for (mlir::BlockArgument blockArg : func.getBody().getArguments()) {
        if (isEncryptedFunctionParameter(blockArg)) {
          unsigned int width = getEintPrecision(blockArg);

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
