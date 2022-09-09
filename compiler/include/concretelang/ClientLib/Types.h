// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_TYPES_H_
#define CONCRETELANG_CLIENTLIB_TYPES_H_

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <stddef.h>
#include <vector>

namespace concretelang {
namespace clientlib {

template <size_t N> struct MemRefDescriptor {
  uint64_t *allocated;
  uint64_t *aligned;
  size_t offset;
  size_t sizes[N];
  size_t strides[N];
};

using decrypted_scalar_t = std::uint64_t;
using decrypted_tensor_1_t = std::vector<decrypted_scalar_t>;
using decrypted_tensor_2_t = std::vector<decrypted_tensor_1_t>;
using decrypted_tensor_3_t = std::vector<decrypted_tensor_2_t>;

template <size_t Rank> using encrypted_tensor_t = MemRefDescriptor<Rank>;
using encrypted_scalar_t = uint64_t *;
using encrypted_scalars_t = uint64_t *;

// Element types for `TensorData`
enum class ElementType { u64, i64, u32, i32, u16, i16, u8, i8 };

namespace {
// Returns the number of bits for an element type
static constexpr size_t getElementTypeWidth(ElementType t) {
  switch (t) {
  case ElementType::u64:
  case ElementType::i64:
    return 64;
  case ElementType::u32:
  case ElementType::i32:
    return 32;
  case ElementType::u16:
  case ElementType::i16:
    return 16;
  case ElementType::u8:
  case ElementType::i8:
    return 8;
  }
}
} // namespace

// Constants for the element types used for tensors representing
// encrypted data and data after decryption
constexpr ElementType EncryptedScalarElementType = ElementType::u64;
constexpr size_t EncryptedScalarElementWidth =
    getElementTypeWidth(ElementType::u64);

using EncryptedScalarElement = uint64_t;

namespace detail {
namespace TensorData {

// Union used to store the pointer to the actual data of an instance
// of `TensorData`. Values are stored contiguously in memory in a
// `std::vector` whose element type corresponds to the element type of
// the tensor.
union value_vector_union {
  std::vector<uint64_t> *u64;
  std::vector<int64_t> *i64;
  std::vector<uint32_t> *u32;
  std::vector<int32_t> *i32;
  std::vector<uint16_t> *u16;
  std::vector<int16_t> *i16;
  std::vector<uint8_t> *u8;
  std::vector<int8_t> *i8;
};

// Function templates that would go into the class `TensorData`, but
// which need to declared in namespace scope, since specializations of
// templates on the return type cannot be done for member functions as
// per the C++ standard
template <typename T> T begin(union value_vector_union &vec);
template <typename T> T end(union value_vector_union &vec);
template <typename T> T cbegin(union value_vector_union &vec);
template <typename T> T cend(union value_vector_union &vec);
template <typename T> T getElements(union value_vector_union &vec);
template <typename T> T getConstElements(const union value_vector_union &vec);

template <typename T>
T getElementValue(union value_vector_union &vec, size_t idx,
                  ElementType elementType);
template <typename T>
T &getElementReference(union value_vector_union &vec, size_t idx,
                       ElementType elementType);
template <typename T>
T *getElementPointer(union value_vector_union &vec, size_t idx,
                     ElementType elementType);

// Specializations for the above templates
#define TENSORDATA_SPECIALIZE_FOR_ITERATOR(ELTY, SUFFIX)                       \
  template <>                                                                  \
  inline std::vector<ELTY>::iterator begin(union value_vector_union &vec) {    \
    return vec.SUFFIX->begin();                                                \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline std::vector<ELTY>::iterator end(union value_vector_union &vec) {      \
    return vec.SUFFIX->end();                                                  \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline std::vector<ELTY>::const_iterator cbegin(                             \
      union value_vector_union &vec) {                                         \
    return vec.SUFFIX->cbegin();                                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline std::vector<ELTY>::const_iterator cend(                               \
      union value_vector_union &vec) {                                         \
    return vec.SUFFIX->cend();                                                 \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline std::vector<ELTY> &getElements(union value_vector_union &vec) {       \
    return *vec.SUFFIX;                                                        \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline const std::vector<ELTY> &getConstElements(                            \
      const union value_vector_union &vec) {                                   \
    return *vec.SUFFIX;                                                        \
  }

TENSORDATA_SPECIALIZE_FOR_ITERATOR(uint64_t, u64)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(int64_t, i64)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(uint32_t, u32)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(int32_t, i32)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(uint16_t, u16)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(int16_t, i16)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(uint8_t, u8)
TENSORDATA_SPECIALIZE_FOR_ITERATOR(int8_t, i8)

#define TENSORDATA_SPECIALIZE_VALUE_GETTER(ELTY, SUFFIX)                       \
  template <>                                                                  \
  inline ELTY getElementValue(union value_vector_union &vec, size_t idx,       \
                              ElementType elementType) {                       \
    assert(elementType == ElementType::SUFFIX);                                \
    return (*vec.SUFFIX)[idx];                                                 \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline ELTY &getElementReference(union value_vector_union &vec, size_t idx,  \
                                   ElementType elementType) {                  \
    assert(elementType == ElementType::SUFFIX);                                \
    return (*vec.SUFFIX)[idx];                                                 \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline ELTY *getElementPointer(union value_vector_union &vec, size_t idx,    \
                                 ElementType elementType) {                    \
    assert(elementType == ElementType::SUFFIX);                                \
    return &(*vec.SUFFIX)[idx];                                                \
  }

TENSORDATA_SPECIALIZE_VALUE_GETTER(uint64_t, u64)
TENSORDATA_SPECIALIZE_VALUE_GETTER(int64_t, i64)
TENSORDATA_SPECIALIZE_VALUE_GETTER(uint32_t, u32)
TENSORDATA_SPECIALIZE_VALUE_GETTER(int32_t, i32)
TENSORDATA_SPECIALIZE_VALUE_GETTER(uint16_t, u16)
TENSORDATA_SPECIALIZE_VALUE_GETTER(int16_t, i16)
TENSORDATA_SPECIALIZE_VALUE_GETTER(uint8_t, u8)
TENSORDATA_SPECIALIZE_VALUE_GETTER(int8_t, i8)

} // namespace TensorData
} // namespace detail

// Representation of a tensor with an arbitrary number of dimensions
class TensorData {
protected:
  detail::TensorData::value_vector_union values;
  ElementType elementType;
  std::vector<size_t> dimensions;

  /* Multi-dimensional, uninitialized, but preallocated tensor */
  void initPreallocated(llvm::ArrayRef<size_t> dimensions,
                        ElementType elementType) {
    assert(dimensions.size() != 0);
    this->dimensions.resize(dimensions.size());

    size_t n = getNumElements(dimensions);

    switch (elementType) {
    case ElementType::u64:
      this->values.u64 = new std::vector<uint64_t>(n);
      break;
    case ElementType::i64:
      this->values.i64 = new std::vector<int64_t>(n);
      break;
    case ElementType::u32:
      this->values.u32 = new std::vector<uint32_t>(n);
      break;
    case ElementType::i32:
      this->values.i32 = new std::vector<int32_t>(n);
      break;
    case ElementType::u16:
      this->values.u16 = new std::vector<uint16_t>(n);
      break;
    case ElementType::i16:
      this->values.i16 = new std::vector<int16_t>(n);
      break;
    case ElementType::u8:
      this->values.u8 = new std::vector<uint8_t>(n);
      break;
    case ElementType::i8:
      this->values.i8 = new std::vector<int8_t>(n);
      break;
    }
    this->elementType = elementType;
    std::copy(dimensions.begin(), dimensions.end(), this->dimensions.begin());
  }

  // Creates a vector<size_t> from an ArrayRef<T>
  template <typename T>
  static std::vector<size_t> toDimSpec(llvm::ArrayRef<T> dims) {
    return std::vector<size_t>(dims.begin(), dims.end());
  }

public:
  // Returns the total number of elements of a tensor with the
  // specified dimensions
  template <typename T> static size_t getNumElements(T dimensions) {
    size_t n = 1;
    for (auto dim : dimensions)
      n *= dim;

    return n;
  }

  // Returns the number of bits of an integer capable of storing
  // values with up to `elementWidth` bits.
  static size_t storageWidth(size_t elementWidth) {
    if (elementWidth > 64) {
      assert(false && "Maximum supported element width is 64");
    } else if (elementWidth > 32) {
      return 64;
    } else if (elementWidth > 16) {
      return 32;
    } else if (elementWidth > 8) {
      return 16;
    } else {
      return 8;
    }
  }

  // Move constructor. Leaves `that` uninitialized.
  TensorData(TensorData &&that)
      : elementType(that.elementType), dimensions(std::move(that.dimensions)) {
    switch (that.elementType) {
    case ElementType::u64:
      this->values.u64 = that.values.u64;
      that.values.u64 = nullptr;
      break;
    case ElementType::i64:
      this->values.i64 = that.values.i64;
      that.values.i64 = nullptr;
      break;
    case ElementType::u32:
      this->values.u32 = that.values.u32;
      that.values.u32 = nullptr;
      break;
    case ElementType::i32:
      this->values.i32 = that.values.i32;
      that.values.i32 = nullptr;
      break;
    case ElementType::u16:
      this->values.u16 = that.values.u16;
      that.values.u16 = nullptr;
      break;
    case ElementType::i16:
      this->values.i16 = that.values.i16;
      that.values.i16 = nullptr;
      break;
    case ElementType::u8:
      this->values.u8 = that.values.u8;
      that.values.u8 = nullptr;
      break;
    case ElementType::i8:
      this->values.i8 = that.values.i8;
      that.values.i8 = nullptr;
      break;
    }
  }

  // Constructor to build a multi-dimensional tensor with the
  // corresponding element type. All elements are initialized with the
  // default value of `0`.
  TensorData(llvm::ArrayRef<size_t> dimensions, ElementType elementType) {
    initPreallocated(dimensions, elementType);
  }

  TensorData(llvm::ArrayRef<int64_t> dimensions, ElementType elementType)
      : TensorData(toDimSpec(dimensions), elementType) {}

  // Constructor to build a multi-dimensional tensor with the element
  // type corresponding to `elementWidth` and `sign`. The value for
  // `elementWidth` must be a power of 2 of up to 64. All elements are
  // initialized with the default value of `0`.
  TensorData(llvm::ArrayRef<size_t> dimensions, size_t elementWidth,
             bool sign) {
    switch (elementWidth) {
    case 64:
      initPreallocated(dimensions,
                       (sign) ? ElementType::i64 : ElementType::u64);
      break;
    case 32:
      initPreallocated(dimensions,
                       (sign) ? ElementType::i32 : ElementType::u32);
      break;
    case 16:
      initPreallocated(dimensions,
                       (sign) ? ElementType::i16 : ElementType::u16);
      break;
    case 8:
      initPreallocated(dimensions, (sign) ? ElementType::i8 : ElementType::u8);
      break;
    default:
      assert(false && "Element width must be 64, 32, 16 or 8 bits");
    }
  }

  TensorData(llvm::ArrayRef<int64_t> dimensions, size_t elementWidth, bool sign)
      : TensorData(toDimSpec(dimensions), elementWidth, sign) {}

#define DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(ELTY, SUFFIX)                       \
  /* Multi-dimensional, initialized tensor, values copied from */              \
  /* `values` */                                                               \
  TensorData(llvm::ArrayRef<ELTY> values, llvm::ArrayRef<size_t> dimensions)   \
      : dimensions(dimensions.begin(), dimensions.end()) {                     \
    assert(dimensions.size() != 0);                                            \
    size_t n = getNumElements(dimensions);                                     \
    this->values.SUFFIX = new std::vector<ELTY>(n);                            \
    this->elementType = ElementType::SUFFIX;                                   \
    this->bulkAssign(values);                                                  \
  }                                                                            \
                                                                               \
  /* One-dimensional, initialized tensor. Values are copied from */            \
  /* `values` */                                                               \
  TensorData(llvm::ArrayRef<ELTY> values)                                      \
      : TensorData(values, llvm::SmallVector<size_t, 1>{values.size()}) {}

  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(uint64_t, u64)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(int64_t, i64)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(uint32_t, u32)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(int32_t, i32)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(uint16_t, u16)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(int16_t, i16)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(uint8_t, u8)
  DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(int8_t, i8)

  ~TensorData() {
    switch (this->elementType) {
    case ElementType::u64:
      delete values.u64;
      break;
    case ElementType::i64:
      delete values.i64;
      break;
    case ElementType::u32:
      delete values.u32;
      break;
    case ElementType::i32:
      delete values.i32;
      break;
    case ElementType::u16:
      delete values.u16;
      break;
    case ElementType::i16:
      delete values.i16;
      break;
    case ElementType::u8:
      delete values.u8;
      break;
    case ElementType::i8:
      delete values.i8;
      break;
    }
  }

  // Returns the total number of elements of the tensor
  size_t length() const { return getNumElements(this->dimensions); }

  // Returns a vector with the size for each dimension of the tensor
  const std::vector<size_t> &getDimensions() const { return this->dimensions; }

  // Returns the number of dimensions
  size_t getRank() const { return this->dimensions.size(); }

  // Multi-dimensional access to a tensor element
  template <typename T> T &operator[](llvm::ArrayRef<int64_t> index) {
    // Number of dimensions must match
    assert(index.size() == dimensions.size());

    int64_t offset = 0;
    int64_t multiplier = 1;
    for (int64_t i = index.size() - 1; i > 0; i--) {
      offset += index[i] * multiplier;
      multiplier *= this->dimensions[i];
    }

    return detail::TensorData::getElementReference<T>(values, offset,
                                                      elementType);
  }

  // Iterator pointing to the first element of a flat representation
  // of the tensor.
  template <typename T> typename std::vector<T>::iterator begin() {
    return detail::TensorData::begin<typename std::vector<T>::iterator>(values);
  }

  // Iterator pointing past the last element of a flat representation
  // of the tensor.
  template <typename T> typename std::vector<T>::iterator end() {
    return detail::TensorData::end<typename std::vector<T>::iterator>(values);
  }

  // Const iterator pointing to the first element of a flat
  // representation of the tensor.
  template <typename T> typename std::vector<T>::iterator cbegin() {
    return detail::TensorData::cbegin<typename std::vector<T>::iterator>(
        values);
  }

  // Const iterator pointing past the last element of a flat
  // representation of the tensor.
  template <typename T> typename std::vector<T>::iterator cend() {
    return detail::TensorData::cend<typename std::vector<T>::iterator>(values);
  }

  // Flat representation of the const tensor
  template <typename T> const std::vector<T> &getElements() const {
    return detail::TensorData::getConstElements<const std::vector<T> &>(values);
  }

  // Flat representation of the tensor
  template <typename T> const std::vector<T> &getElements() {
    return detail::TensorData::getElements<std::vector<T> &>(values);
  }

  // Returns the `index`-th value of a flat representation of the tensor
  template <typename T> T getElementValue(size_t index) {
    return detail::TensorData::getElementValue<T>(values, index, elementType);
  }

  // Returns a reference to the `index`-th value of a flat
  // representation of the tensor
  template <typename T> T &getElementReference(size_t index) {
    return detail::TensorData::getElementReference<T>(values, index,
                                                      elementType);
  }

  // Returns a pointer to the `index`-th value of a flat
  // representation of the tensor
  template <typename T> T *getElementPointer(size_t index) {
    return detail::TensorData::getElementPointer<T>(values, index, elementType);
  }

  // Returns a void pointer to the `index`-th value of a flat
  // representation of the tensor
  void *getOpaqueElementPointer(size_t index) {
    switch (this->elementType) {
    case ElementType::u64:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<uint64_t>(values, index,
                                                          elementType));
    case ElementType::i64:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<int64_t>(values, index,
                                                         elementType));
    case ElementType::u32:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<uint32_t>(values, index,
                                                          elementType));
    case ElementType::i32:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<int32_t>(values, index,
                                                         elementType));
    case ElementType::u16:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<uint16_t>(values, index,
                                                          elementType));
    case ElementType::i16:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<int16_t>(values, index,
                                                         elementType));
    case ElementType::u8:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<uint8_t>(values, index,
                                                         elementType));
    case ElementType::i8:
      return reinterpret_cast<void *>(
          detail::TensorData::getElementPointer<int8_t>(values, index,
                                                        elementType));
    }

    assert(false && "Unknown element type");
  }

  // Returns the element type of the tensor
  ElementType getElementType() const { return this->elementType; }

  // Returns the size of a tensor element in bytes
  size_t getElementSize() const {
    switch (this->elementType) {
    case ElementType::u64:
    case ElementType::i64:
      return 8;
    case ElementType::u32:
    case ElementType::i32:
      return 4;
    case ElementType::u16:
    case ElementType::i16:
      return 2;
    case ElementType::u8:
    case ElementType::i8:
      return 1;
    }
  }

  // Returns `true` if elements are signed, otherwise `false`
  bool getElementSignedness() const {
    switch (this->elementType) {
    case ElementType::u64:
    case ElementType::u32:
    case ElementType::u16:
    case ElementType::u8:
      return false;
    case ElementType::i64:
    case ElementType::i32:
    case ElementType::i16:
    case ElementType::i8:
      return true;
    }
  }

  // Returns the width of an element in bits
  size_t getElementWidth() const {
    return getElementTypeWidth(this->elementType);
  }

  // Returns the total number of elements of the tensor
  size_t getNumElements() const { return getNumElements(this->dimensions); }

  // Copy all elements from `values` to the tensor. Note that this
  // does not append values to the tensor, but overwrites existing
  // values.
  template <typename T> void bulkAssign(llvm::ArrayRef<T> values) {
    assert(values.size() <= this->getNumElements());

    switch (this->elementType) {
    case ElementType::u64:
      std::copy(values.begin(), values.end(), this->values.u64->begin());
      break;
    case ElementType::i64:
      std::copy(values.begin(), values.end(), this->values.i64->begin());
      break;
    case ElementType::u32:
      std::copy(values.begin(), values.end(), this->values.u32->begin());
      break;
    case ElementType::i32:
      std::copy(values.begin(), values.end(), this->values.i32->begin());
      break;
    case ElementType::u16:
      std::copy(values.begin(), values.end(), this->values.u16->begin());
      break;
    case ElementType::i16:
      std::copy(values.begin(), values.end(), this->values.i16->begin());
      break;
    case ElementType::u8:
      std::copy(values.begin(), values.end(), this->values.u8->begin());
      break;
    case ElementType::i8:
      std::copy(values.begin(), values.end(), this->values.i8->begin());
      break;
    }
  }

  // Copies all elements of a flat representation of the tensor to the
  // positions starting with the iterator `start`.
  template <typename IT> void copy(IT start) {
    switch (this->elementType) {
    case ElementType::u64:
      std::copy(this->values.u64->begin(), this->values.u64->end(), start);
      break;
    case ElementType::i64:
      std::copy(this->values.i64->begin(), this->values.i64->end(), start);
      break;
    case ElementType::u32:
      std::copy(this->values.u32->begin(), this->values.u32->end(), start);
      break;
    case ElementType::i32:
      std::copy(this->values.i32->begin(), this->values.i32->end(), start);
      break;
    case ElementType::u16:
      std::copy(this->values.u16->begin(), this->values.u16->end(), start);
      break;
    case ElementType::i16:
      std::copy(this->values.i16->begin(), this->values.i16->end(), start);
      break;
    case ElementType::u8:
      std::copy(this->values.u8->begin(), this->values.u8->end(), start);
      break;
    case ElementType::i8:
      std::copy(this->values.i8->begin(), this->values.i8->end(), start);
      break;
    }
  }

  // Returns a flat representation of the tensor with elements
  // converted to the type `T`
  template <typename T> std::vector<T> asFlatVector() {
    std::vector<T> ret(getNumElements());
    this->copy(ret.begin());
    return ret;
  }

  // Returns a void pointer to the first element of a flat
  // representation of the tensor
  void *getValuesAsOpaquePointer() {
    switch (this->elementType) {
    case ElementType::u64:
      return static_cast<void *>(values.u64->data());
    case ElementType::i64:
      return static_cast<void *>(values.i64->data());
    case ElementType::u32:
      return static_cast<void *>(values.u32->data());
    case ElementType::i32:
      return static_cast<void *>(values.i32->data());
    case ElementType::u16:
      return static_cast<void *>(values.u16->data());
    case ElementType::i16:
      return static_cast<void *>(values.i16->data());
    case ElementType::u8:
      return static_cast<void *>(values.u8->data());
    case ElementType::i8:
      return static_cast<void *>(values.i8->data());
    }

    assert(false && "Unhandled element type");
  }
};

} // namespace clientlib
} // namespace concretelang

#endif
