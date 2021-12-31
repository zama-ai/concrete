// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_SUPPORT_LAMBDA_ARGUMENT_H
#define CONCRETELANG_SUPPORT_LAMBDA_ARGUMENT_H

#include <cstdint>
#include <limits>

#include <concretelang/Support/Error.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ExtensibleRTTI.h>

namespace mlir {
namespace concretelang {

// Abstract base class for lambda arguments
class LambdaArgument
    : public llvm::RTTIExtends<LambdaArgument, llvm::RTTIRoot> {
public:
  LambdaArgument(LambdaArgument &) = delete;

  template <typename T> bool isa() const { return llvm::isa<T>(*this); }

  // Cast functions on constant instances
  template <typename T> const T &cast() const { return llvm::cast<T>(*this); }
  template <typename T> const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  // Cast functions for mutable instances
  template <typename T> T &cast() { return llvm::cast<T>(*this); }
  template <typename T> T *dyn_cast() { return llvm::dyn_cast<T>(this); }

  static char ID;

protected:
  LambdaArgument(){};
};

// Class for integer arguments. `BackingIntType` is used as the data
// type to hold the argument's value. The precision is the actual
// precision of the value, which might be different from the precision
// of the backing integer type.
template <typename BackingIntType = uint64_t>
class IntLambdaArgument
    : public llvm::RTTIExtends<IntLambdaArgument<BackingIntType>,
                               LambdaArgument> {
public:
  typedef BackingIntType value_type;

  IntLambdaArgument(BackingIntType value,
                    unsigned int precision = 8 * sizeof(BackingIntType))
      : precision(precision) {
    if (precision < 8 * sizeof(BackingIntType)) {
      this->value = value & (1 << (this->precision - 1));
    } else {
      this->value = value;
    }
  }

  unsigned int getPrecision() const { return this->precision; }
  BackingIntType getValue() const { return this->value; }

  static char ID;

protected:
  unsigned int precision;
  BackingIntType value;
};

template <typename BackingIntType>
char IntLambdaArgument<BackingIntType>::ID = 0;

// Class for encrypted integer arguments. `BackingIntType` is used as
// the data type to hold the argument's plaintext value. The precision
// is the actual precision of the value, which might be different from
// the precision of the backing integer type.
template <typename BackingIntType = uint64_t>
class EIntLambdaArgument
    : public llvm::RTTIExtends<EIntLambdaArgument<BackingIntType>,
                               IntLambdaArgument<BackingIntType>> {
public:
  static char ID;
};

template <typename BackingIntType>
char EIntLambdaArgument<BackingIntType>::ID = 0;

namespace {
// Calculates `accu *= factor` or returns an error if the result
// would overflow
template <typename AccuT, typename ValT>
llvm::Error safeUnsignedMul(AccuT &accu, ValT factor) {
  static_assert(std::numeric_limits<AccuT>::is_integer &&
                    std::numeric_limits<ValT>::is_integer &&
                    !std::numeric_limits<AccuT>::is_signed &&
                    !std::numeric_limits<ValT>::is_signed,
                "Only unsigned integers are supported");

  const AccuT left = std::numeric_limits<AccuT>::max() / accu;

  if (left > factor) {
    accu *= factor;
    return llvm::Error::success();
  }

  return StreamStringError("Multiplying value ")
         << accu << " with " << factor << " would cause an overflow";
}
} // namespace

// Class for Tensor arguments. This can either be plaintext tensors
// (for `ScalarArgumentT = IntLambaArgument<T>`) or tensors
// representing encrypted integers (for `ScalarArgumentT =
// EIntLambaArgument<T>`).
template <typename ScalarArgumentT>
class TensorLambdaArgument
    : public llvm::RTTIExtends<TensorLambdaArgument<ScalarArgumentT>,
                               LambdaArgument> {
public:
  typedef ScalarArgumentT scalar_type;

  // Construct tensor argument from the one-dimensional array `value`,
  // but interpreting the array's values as a linearized
  // multi-dimensional tensor with the sizes of the dimensions
  // specified in `dimensions`.
  TensorLambdaArgument(
      llvm::ArrayRef<typename ScalarArgumentT::value_type> value,
      llvm::ArrayRef<int64_t> dimensions)
      : dimensions(dimensions.vec()) {
    std::copy(value.begin(), value.end(), std::back_inserter(this->value));
  }

  // Construct a one-dimensional tensor argument from the
  // array `value`.
  TensorLambdaArgument(
      llvm::ArrayRef<typename ScalarArgumentT::value_type> value)
      : TensorLambdaArgument(value, {(int64_t)value.size()}) {}

  template <std::size_t size1, std::size_t size2>
  TensorLambdaArgument(
      typename ScalarArgumentT::value_type (&a)[size1][size2]) {
    dimensions = {size1, size2};
    auto value = llvm::MutableArrayRef<typename ScalarArgumentT::value_type>(
        (typename ScalarArgumentT::value_type *)a, size1 * size2);
    std::copy(value.begin(), value.end(), std::back_inserter(this->value));
  }

  const std::vector<int64_t> &getDimensions() const { return this->dimensions; }

  // Returns the total number of elements in the tensor. If the number
  // of elements cannot be represented as a `size_t`, the method
  // returns an error.
  llvm::Expected<size_t> getNumElements() const {
    size_t accu = 1;

    for (unsigned int dimSize : dimensions)
      if (llvm::Error err = safeUnsignedMul(accu, dimSize))
        return std::move(err);

    return accu;
  }

  // Returns a bare pointer to the linearized values of the tensor
  // (constant version).
  const typename ScalarArgumentT::value_type *getValue() const {
    return this->value.data();
  }

  // Returns a bare pointer to the linearized values of the tensor (mutable
  // version).
  typename ScalarArgumentT::value_type *getValue() {
    return this->value.data();
  }

  static char ID;

protected:
  std::vector<typename ScalarArgumentT::value_type> value;
  std::vector<int64_t> dimensions;
};

template <typename ScalarArgumentT>
char TensorLambdaArgument<ScalarArgumentT>::ID = 0;

} // namespace concretelang
} // namespace mlir

#endif
