// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_TYPES_H_
#define CONCRETELANG_CLIENTLIB_TYPES_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <stddef.h>
#include <variant>
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

// Returns the width in bits of an integer whose width is a power of
// two that can hold values with at most `width` bits
static inline constexpr size_t getStorageWidth(size_t width) {
  if (width > 64)
    assert(false && "Unsupported scalar width");

  if (width > 32) {
    return 64;
  } else if (width > 16) {
    return 32;
  } else if (width > 8) {
    return 16;
  } else {
    return 8;
  }
}

// Translates `sign` and `width` into an `ElementType`.
static inline ElementType getElementTypeFromWidthAndSign(size_t width,
                                                         bool sign) {
  switch (getStorageWidth(width)) {
  case 64:
    return (sign) ? ElementType::i64 : ElementType::u64;
  case 32:
    return (sign) ? ElementType::i32 : ElementType::u32;
  case 16:
    return (sign) ? ElementType::i16 : ElementType::u16;
  case 8:
    return (sign) ? ElementType::i8 : ElementType::u8;
  default:
    assert(false && "Unsupported scalar width");
  }
}

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

  // Cannot happen
  return 0;
}

// Returns `true` if the element type `t` designates a signed type,
// otherwise `false`.
static constexpr size_t getElementTypeSignedness(ElementType t) {
  switch (t) {
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

  // Cannot happen
  return false;
}

// Returns `true` iff the element type `t` designates the smallest
// unsigned / signed (depending on `sign`) integer type that can hold
// values of up to `width` bits, otherwise false.
static inline bool checkElementTypeForWidthAndSign(ElementType t, size_t width,
                                                   bool sign) {
  return getElementTypeFromWidthAndSign(getStorageWidth(width), sign) == t;
}
} // namespace

// Constants for the element types used for tensors representing
// encrypted data and data after decryption
constexpr ElementType EncryptedScalarElementType = ElementType::u64;
constexpr size_t EncryptedScalarElementWidth =
    getElementTypeWidth(EncryptedScalarElementType);

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
  size_t elementWidth;

  /* Multi-dimensional, uninitialized, but preallocated tensor */
  void initPreallocated(llvm::ArrayRef<size_t> dimensions,
                        ElementType elementType, size_t elementWidth,
                        bool sign) {
    assert(checkElementTypeForWidthAndSign(elementType, elementWidth, sign) &&
           "Incoherent parameters for element type, width and sign");

    assert(dimensions.size() != 0);

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

    this->dimensions.resize(dimensions.size());
    this->elementWidth = elementWidth;
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

  // Move constructor. Leaves `that` uninitialized.
  TensorData(TensorData &&that)
      : elementType(that.elementType), dimensions(std::move(that.dimensions)),
        elementWidth(that.elementWidth) {
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
  TensorData(llvm::ArrayRef<size_t> dimensions, ElementType elementType,
             size_t elementWidth) {
    initPreallocated(dimensions, elementType, elementWidth,
                     getElementTypeSignedness(elementType));
  }

  TensorData(llvm::ArrayRef<int64_t> dimensions, ElementType elementType,
             size_t elementWidth)
      : TensorData(toDimSpec(dimensions), elementType, elementWidth) {}

  // Constructor to build a multi-dimensional tensor with the element
  // type corresponding to `elementWidth` and `sign`. All elements are
  // initialized with the default value of `0`.
  TensorData(llvm::ArrayRef<size_t> dimensions, size_t elementWidth, bool sign)
      : TensorData(dimensions,
                   getElementTypeFromWidthAndSign(elementWidth, sign),
                   elementWidth) {}

  TensorData(llvm::ArrayRef<int64_t> dimensions, size_t elementWidth, bool sign)
      : TensorData(toDimSpec(dimensions), elementWidth, sign) {}

#define DEF_TENSOR_DATA_TENSOR_COSTRUCTORS(ELTY, SUFFIX)                       \
  /* Multi-dimensional, initialized tensor, values copied from */              \
  /* `values` */                                                               \
  TensorData(llvm::ArrayRef<ELTY> values, llvm::ArrayRef<size_t> dimensions,   \
             size_t elementWidth)                                              \
      : dimensions(dimensions.begin(), dimensions.end()) {                     \
    assert(checkElementTypeForWidthAndSign(ElementType::SUFFIX, elementWidth,  \
                                           std::is_signed<ELTY>()) &&          \
           "wrong element type for width");                                    \
    assert(dimensions.size() != 0);                                            \
    size_t n = getNumElements(dimensions);                                     \
    this->values.SUFFIX = new std::vector<ELTY>(n);                            \
    this->elementType = ElementType::SUFFIX;                                   \
    this->bulkAssign(values);                                                  \
  }                                                                            \
                                                                               \
  /* One-dimensional, initialized tensor. Values are copied from */            \
  /* `values` */                                                               \
  TensorData(llvm::ArrayRef<ELTY> values, size_t width)                        \
      : TensorData(values, llvm::SmallVector<size_t, 1>{values.size()},        \
                   width) {}

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

  template <typename T> const std::vector<T> getDimensionsAs() const {
    return std::vector<T>(this->dimensions.begin(), this->dimensions.end());
  }

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

  // Returns a pointer to the `index`-th value of a flat
  // representation of the tensor (const version)
  template <typename T> const T *getElementPointer(size_t index) const {
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

  // Returns the actual width in bits of a data element (i.e., the
  // width specified upon construction and not the storage width of an
  // element)
  size_t getElementWidth() const { return this->elementWidth; }

  // Returns the size of a tensor element in bytes (i.e., the storage width in
  // bytes)
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
  template <typename IT> void copy(IT start) const {
    switch (this->elementType) {
    case ElementType::u64:
      std::copy(this->values.u64->cbegin(), this->values.u64->cend(), start);
      break;
    case ElementType::i64:
      std::copy(this->values.i64->cbegin(), this->values.i64->cend(), start);
      break;
    case ElementType::u32:
      std::copy(this->values.u32->cbegin(), this->values.u32->cend(), start);
      break;
    case ElementType::i32:
      std::copy(this->values.i32->cbegin(), this->values.i32->cend(), start);
      break;
    case ElementType::u16:
      std::copy(this->values.u16->cbegin(), this->values.u16->cend(), start);
      break;
    case ElementType::i16:
      std::copy(this->values.i16->cbegin(), this->values.i16->cend(), start);
      break;
    case ElementType::u8:
      std::copy(this->values.u8->cbegin(), this->values.u8->cend(), start);
      break;
    case ElementType::i8:
      std::copy(this->values.i8->cbegin(), this->values.i8->cend(), start);
      break;
    }
  }

  // Returns a flat representation of the tensor with elements
  // converted to the type `T`
  template <typename T> std::vector<T> asFlatVector() const {
    std::vector<T> ret(getNumElements());
    this->copy(ret.begin());
    return ret;
  }

  // Returns a void pointer to the first element of a flat
  // representation of the tensor
  void *getValuesAsOpaquePointer() const {
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

namespace detail {
namespace ScalarData {
// Union representing a single scalar value
union scalar_union {
  uint64_t u64;
  int64_t i64;
  uint32_t u32;
  int32_t i32;
  uint16_t u16;
  int16_t i16;
  uint8_t u8;
  int8_t i8;
};

// Template + specializations that should be in ScalarData, but which need to be
// in namespace scope
template <typename T> T getValue(const union scalar_union &u, ElementType type);

#define SCALARDATA_SPECIALIZE_VALUE_GETTER(ELTY, SUFFIX)                       \
  template <>                                                                  \
  inline ELTY getValue(const union scalar_union &u, ElementType type) {        \
    assert(type == ElementType::SUFFIX);                                       \
    return u.SUFFIX;                                                           \
  }

SCALARDATA_SPECIALIZE_VALUE_GETTER(uint64_t, u64)
SCALARDATA_SPECIALIZE_VALUE_GETTER(int64_t, i64)
SCALARDATA_SPECIALIZE_VALUE_GETTER(uint32_t, u32)
SCALARDATA_SPECIALIZE_VALUE_GETTER(int32_t, i32)
SCALARDATA_SPECIALIZE_VALUE_GETTER(uint16_t, u16)
SCALARDATA_SPECIALIZE_VALUE_GETTER(int16_t, i16)
SCALARDATA_SPECIALIZE_VALUE_GETTER(uint8_t, u8)
SCALARDATA_SPECIALIZE_VALUE_GETTER(int8_t, i8)

} // namespace ScalarData
} // namespace detail

// Class representing a single scalar value
class ScalarData {
public:
  ScalarData(const ScalarData &s)
      : type(s.type), value(s.value), width(s.width) {}

  // Construction with a specific type and an actual width, but with a value
  // provided in a generic `uint64_t`
  ScalarData(uint64_t value, ElementType type, size_t width)
      : type(type), width(width) {
    assert(width <= getElementTypeWidth(type));

    switch (type) {
    case ElementType::u64:
      this->value.u64 = value;
      break;
    case ElementType::i64:
      this->value.i64 = value;
      break;
    case ElementType::u32:
      this->value.u32 = value;
      break;
    case ElementType::i32:
      this->value.i32 = value;
      break;
    case ElementType::u16:
      this->value.u16 = value;
      break;
    case ElementType::i16:
      this->value.i16 = value;
      break;
    case ElementType::u8:
      this->value.u8 = value;
      break;
    case ElementType::i8:
      this->value.i8 = value;
      break;
    }
  }

  // Construction with a specific type determined by `sign` and
  // `width`, but value provided in a generic `uint64_t`
  ScalarData(uint64_t value, bool sign, size_t width)
      : ScalarData(value, getElementTypeFromWidthAndSign(width, sign), width) {}

#define DEF_SCALAR_DATA_CONSTRUCTOR(ELTY, SUFFIX)                              \
  ScalarData(ELTY value)                                                       \
      : type(ElementType::SUFFIX),                                             \
        width(getElementTypeWidth(ElementType::SUFFIX)) {                      \
    this->value.SUFFIX = value;                                                \
  }

  // Construction from specific value type
  DEF_SCALAR_DATA_CONSTRUCTOR(uint64_t, u64)
  DEF_SCALAR_DATA_CONSTRUCTOR(int64_t, i64)
  DEF_SCALAR_DATA_CONSTRUCTOR(uint32_t, u32)
  DEF_SCALAR_DATA_CONSTRUCTOR(int32_t, i32)
  DEF_SCALAR_DATA_CONSTRUCTOR(uint16_t, u16)
  DEF_SCALAR_DATA_CONSTRUCTOR(int16_t, i16)
  DEF_SCALAR_DATA_CONSTRUCTOR(uint8_t, u8)
  DEF_SCALAR_DATA_CONSTRUCTOR(int8_t, i8)

  template <typename T> T getValue() const {
    return detail::ScalarData::getValue<T>(value, type);
  }

  // Retrieves the value as a generic `uint64_t`
  uint64_t getValueAsU64() const {
    size_t width = getElementTypeWidth(type);
    if (width == 64)
      return value.u64;
    uint64_t mask = ((uint64_t)1 << width) - 1;
    uint64_t val = value.u64 & mask;
    return val;
  }

  ElementType getType() const { return type; }
  size_t getWidth() const { return width; }

protected:
  ElementType type;
  union detail::ScalarData::scalar_union value;
  size_t width;
};

typedef std::variant<ScalarData, TensorData> ScalarOrTensorData;

struct SharedScalarOrTensorData {
  std::shared_ptr<ScalarOrTensorData> inner;

  SharedScalarOrTensorData(std::shared_ptr<ScalarOrTensorData> inner)
      : inner{inner} {}

  SharedScalarOrTensorData(ScalarOrTensorData &&inner)
      : inner{std::make_shared<ScalarOrTensorData>(std::move(inner))} {}

  ScalarOrTensorData &get() const { return *this->inner; }
};

} // namespace clientlib
} // namespace concretelang

#endif
