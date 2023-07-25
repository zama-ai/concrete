// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_VALUES_H
#define CONCRETELANG_COMMON_VALUES_H

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Protocol.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdlib.h>
#include <variant>

using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::protocol::dimensionsToProtoShape;
using concretelang::protocol::protoDataToVector;
using concretelang::protocol::protoShapeToDimensions;
using concretelang::protocol::vectorToProtoData;

namespace concretelang {
namespace values {

/// A type for public (encrypted or not) values, that can be safely transported
/// between client and server to for execution.
typedef concreteprotocol::Value TransportValue;

/// A type for tensor data.
template <typename T> struct Tensor {
  std::vector<T> values;
  std::vector<size_t> dimensions;

  Tensor<T>() = default;
  Tensor<T>(std::vector<T> values, std::vector<size_t> dimensions)
      : values(values), dimensions(dimensions) {}

  /// Creates an tensor with the shape described by the input dimensions, filled
  /// with zeros.
  static Tensor<T> fromDimensions(std::vector<size_t> &dimensions) {
    uint32_t length = 1;
    for (auto dim : dimensions) {
      length *= dim;
    }
    auto values = std::vector<T>(length);
    for (auto &val : values) {
      *val = 0;
    }
    return Tensor{values, dimensions};
  }

  /// Conversion constructor from a scalar value.
  Tensor<T>(T in) { this->values.push_back(in); }

  /// Constructor from initializer lists of values and dimensions.
  Tensor<T>(std::initializer_list<T> values,
            std::initializer_list<size_t> dimensions) {
    size_t count = 1;
    for (auto dim : dimensions) {
      count *= dim;
    }
    assert(values.size() == count);
    for (auto val : values) {
      this->values.push_back(val);
    }
    for (auto dim : dimensions) {
      this->dimensions.push_back(dim);
    }
  }
  
  bool operator==(const Tensor<T>& b) const{
    return this->values == b.values && this->dimensions == b.dimensions;
  }

  Tensor<T> operator-(T b) const{
    Tensor<T> out = *this;
    for(size_t i = 0; i<out.values.size(); i++){
      out.values[i] -= b;
    }
    return out;
  }

  Tensor<T> operator-(Tensor<T> b) const{
    assert(this->dimensions == b.dimensions);
    Tensor<T> out = *this;
    for(size_t i = 0; i<out.values.size(); i++){
      out.values[i] -= b.values[i];
    }
    return out;
  }

  Tensor<T> operator+(T b) const{
    Tensor<T> out = *this;
    for(size_t i = 0; i<out.values.size(); i++){
      out.values[i] += b;
    }
    return out;
  }

  Tensor<T> operator+(Tensor<T> b) const{
    assert(this->dimensions == b.dimensions);
    Tensor<T> out = *this;
    for(size_t i = 0; i<out.values.size(); i++){
      out.values[i] += b.values[i];
    }
    return out;
  }

  Tensor<T> operator*(T b) const{
    Tensor<T> out = *this;
    for(size_t i = 0; i<out.values.size(); i++){
      out.values[i] *= b;
    }
    return out;
  }

  Tensor<T> operator*(Tensor<T> b) const{
    assert(this->dimensions == b.dimensions);
    Tensor<T> out = *this;
    for(size_t i = 0; i<out.values.size(); i++){
      out.values[i] *= b.values[i];
    }
    return out;
  }
  
  T& operator[](int index) {
    return this->values[index];
  }

  bool isScalar() const { return dimensions.empty(); }
};

/// A type for tensor data of varying precisions. Mainly use to manipulate
struct Value {
  friend class ClientCircuit;

  std::variant<Tensor<uint8_t>, Tensor<int8_t>, Tensor<uint16_t>,
               Tensor<int16_t>, Tensor<uint32_t>, Tensor<int32_t>,
               Tensor<uint64_t>, Tensor<int64_t>>
      inner;
  Value() = default;
  Value(Tensor<uint8_t> inner) : inner(inner){};
  Value(Tensor<uint16_t> inner) : inner(inner){};
  Value(Tensor<uint32_t> inner) : inner(inner){};
  Value(Tensor<uint64_t> inner) : inner(inner){};
  Value(Tensor<int8_t> inner) : inner(inner){};
  Value(Tensor<int16_t> inner) : inner(inner){};
  Value(Tensor<int32_t> inner) : inner(inner){};
  Value(Tensor<int64_t> inner) : inner(inner){};

  /// Turns a server value to a client value, without interpreting the kind of
  /// value.
  static Value fromRawTransportValue(const TransportValue &transportVal);

  /// Turns a client value to a raw (without kind info attached) server value.
  TransportValue intoRawTransportValue();

  uint32_t getIntegerPrecision();

  uint32_t isSigned();

  std::string *intoProtoData();

  concreteprotocol::Shape *intoProtoShape();

  std::vector<size_t> getDimensions();

  size_t getLength();

  template <typename T> bool hasElementType() const{
    return std::holds_alternative<Tensor<T>>(inner);
  }

  template <typename T> std::optional<Tensor<T>> getTensor() const{
    if (!hasElementType<T>()) {
      return std::nullopt;
    }
    return std::get<Tensor<T>>(inner);
  }

  template <typename T> Tensor<T> *getTensorPtr(){
    if (!hasElementType<T>()) {
      return nullptr;
    }
    return &std::get<Tensor<T>>(inner);
  }

  bool isCompatibleWithShape(const concreteprotocol::Shape &shape);
};

size_t getCorrespondingPrecision(size_t originalPrecision);

} // namespace values
} // namespace concretelang

#endif
