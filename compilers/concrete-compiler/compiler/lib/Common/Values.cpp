// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Values.h"
#include "capnp/common.h"
#include "capnp/list.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Protocol.h"
#include <cstddef>
#include <cstdint>
#include <stdlib.h>
#include <string>

using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::protocol::dimensionsToProtoShape;
using concretelang::protocol::Message;
using concretelang::protocol::protoPayloadToVector;
using concretelang::protocol::protoShapeToDimensions;
using concretelang::protocol::vectorToProtoPayload;

namespace concretelang {
namespace values {

Value Value::fromRawTransportValue(const TransportValue &transportVal) {
  Value output;
  auto integerPrecision =
      transportVal.asReader().getRawInfo().getIntegerPrecision();
  auto isSigned = transportVal.asReader().getRawInfo().getIsSigned();
  auto dimensions =
      protoShapeToDimensions(transportVal.asReader().getRawInfo().getShape());
  auto data = transportVal.asReader().getPayload();
  if (integerPrecision == 8 && isSigned) {
    auto values = protoPayloadToVector<int8_t>(data);
    output.inner = Tensor<int8_t>{values, dimensions};
  } else if (integerPrecision == 16 && isSigned) {
    auto values = protoPayloadToVector<int16_t>(data);
    output.inner = Tensor<int16_t>{values, dimensions};
  } else if (integerPrecision == 32 && isSigned) {
    auto values = protoPayloadToVector<int32_t>(data);
    output.inner = Tensor<int32_t>{values, dimensions};
  } else if (integerPrecision == 64 && isSigned) {
    auto values = protoPayloadToVector<int64_t>(data);
    output.inner = Tensor<int64_t>{values, dimensions};
  } else if (integerPrecision == 8 && !isSigned) {
    auto values = protoPayloadToVector<uint8_t>(data);
    output.inner = Tensor<uint8_t>{values, dimensions};
  } else if (integerPrecision == 16 && !isSigned) {
    auto values = protoPayloadToVector<uint16_t>(data);
    output.inner = Tensor<uint16_t>{values, dimensions};
  } else if (integerPrecision == 32 && !isSigned) {
    auto values = protoPayloadToVector<uint32_t>(data);
    output.inner = Tensor<uint32_t>{values, dimensions};
  } else if (integerPrecision == 64 && !isSigned) {
    auto values = protoPayloadToVector<uint64_t>(data);
    output.inner = Tensor<uint64_t>{values, dimensions};
  } else {
    assert(false);
  }

  return output;
}

TransportValue Value::intoRawTransportValue() const {
  auto output = Message<concreteprotocol::Value>();
  auto rawInfo = output.asBuilder().initRawInfo();
  rawInfo.setShape(intoProtoShape().asReader());
  rawInfo.setIntegerPrecision(getIntegerPrecision());
  rawInfo.setIsSigned(isSigned());
  output.asBuilder().setPayload(intoProtoPayload().asReader());
  return output;
}

uint32_t Value::getIntegerPrecision() const {
  if (hasElementType<uint8_t>() || hasElementType<int8_t>()) {
    return 8;
  } else if (hasElementType<uint16_t>() || hasElementType<int16_t>()) {
    return 16;
  } else if (hasElementType<uint32_t>() || hasElementType<int32_t>()) {
    return 32;
  } else if (hasElementType<uint64_t>() || hasElementType<int64_t>()) {
    return 64;
  } else {
    assert(false);
  }
}

bool Value::isSigned() const {

  if (hasElementType<uint8_t>() || hasElementType<uint16_t>() ||
      hasElementType<uint32_t>() || hasElementType<uint64_t>()) {
    return false;
  } else if (hasElementType<int8_t>() || hasElementType<int16_t>() ||
             hasElementType<int32_t>() || hasElementType<int64_t>()) {
    return true;
  } else {
    assert(false);
  }
}

Message<concreteprotocol::Payload> Value::intoProtoPayload() const {
  if (hasElementType<uint8_t>()) {
    return vectorToProtoPayload(std::get<Tensor<uint8_t>>(inner).values);
  } else if (hasElementType<uint16_t>()) {
    return vectorToProtoPayload(std::get<Tensor<uint16_t>>(inner).values);
  } else if (hasElementType<uint32_t>()) {
    return vectorToProtoPayload(std::get<Tensor<uint32_t>>(inner).values);
  } else if (hasElementType<uint64_t>()) {
    return vectorToProtoPayload(std::get<Tensor<uint64_t>>(inner).values);
  } else if (hasElementType<int8_t>()) {
    return vectorToProtoPayload(std::get<Tensor<int8_t>>(inner).values);
  } else if (hasElementType<int16_t>()) {
    return vectorToProtoPayload(std::get<Tensor<int16_t>>(inner).values);
  } else if (hasElementType<int32_t>()) {
    return vectorToProtoPayload(std::get<Tensor<int32_t>>(inner).values);
  } else if (hasElementType<int64_t>()) {
    return vectorToProtoPayload(std::get<Tensor<int64_t>>(inner).values);
  } else {
    assert(false);
  }
}

Message<concreteprotocol::Shape> Value::intoProtoShape() const {
  return dimensionsToProtoShape(getDimensions());
}

const std::vector<size_t> &Value::getDimensions() const {
  if (hasElementType<uint8_t>()) {
    return std::get<Tensor<uint8_t>>(inner).dimensions;
  }
  if (hasElementType<uint16_t>()) {
    return std::get<Tensor<uint16_t>>(inner).dimensions;
  }
  if (hasElementType<uint32_t>()) {
    return std::get<Tensor<uint32_t>>(inner).dimensions;
  }
  if (hasElementType<uint64_t>()) {
    return std::get<Tensor<uint64_t>>(inner).dimensions;
  }
  if (hasElementType<int8_t>()) {
    return std::get<Tensor<int8_t>>(inner).dimensions;
  }
  if (hasElementType<int16_t>()) {
    return std::get<Tensor<int16_t>>(inner).dimensions;
  }
  if (hasElementType<int32_t>()) {
    return std::get<Tensor<int32_t>>(inner).dimensions;
  }
  if (hasElementType<int64_t>()) {
    return std::get<Tensor<int64_t>>(inner).dimensions;
  }
  assert(false);
}

size_t Value::getLength() const {
  if (auto tensor = getTensor<uint8_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<int8_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return tensor.value().values.size();
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return tensor.value().values.size();
  } else {
    assert(false);
  }
}

bool Value::isCompatibleWithShape(
    const Message<concreteprotocol::Shape> &shape) const {
  return isCompatibleWithShape(shape.asReader());
}

bool Value::isCompatibleWithShape(
    concreteprotocol::Shape::Reader reader) const {
  auto dimensions = getDimensions();
  if ((uint32_t)reader.getDimensions().size() != dimensions.size()) {
    return false;
  }
  for (uint32_t i = 0; i < dimensions.size(); i++) {
    if (reader.getDimensions()[i] != dimensions[i]) {
      return false;
    }
  }
  return true;
}

bool Value::operator==(const Value &b) const {
  if (auto tensor = getTensor<uint8_t>(); tensor) {
    return tensor == b.getTensor<uint8_t>();
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return tensor == b.getTensor<uint16_t>();
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return tensor == b.getTensor<uint32_t>();
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return tensor == b.getTensor<uint64_t>();
  } else if (auto tensor = getTensor<int8_t>(); tensor) {
    return tensor == b.getTensor<int8_t>();
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return tensor == b.getTensor<int16_t>();
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return tensor == b.getTensor<int32_t>();
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return tensor == b.getTensor<int64_t>();
  } else {
    assert(false);
  }
}

bool Value::isScalar() const {
  if (auto tensor = getTensor<int8_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<uint8_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return tensor.value().isScalar();
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return tensor.value().isScalar();
  } else {
    assert(false);
  }
}

Value Value::toUnsigned() const {
  if (!this->isSigned()) {
    return *this;
  } else if (auto tensor = getTensor<int8_t>(); tensor) {
    return Value((Tensor<uint8_t>)tensor.value());
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return Value((Tensor<uint16_t>)tensor.value());
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return Value((Tensor<uint32_t>)tensor.value());
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return Value((Tensor<uint8_t>)tensor.value());
  } else {
    assert(false);
  }
}

Value Value::toSigned() const {
  if (!this->isSigned()) {
    return *this;
  } else if (auto tensor = getTensor<uint8_t>(); tensor) {
    return Value((Tensor<int8_t>)tensor.value());
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return Value((Tensor<int16_t>)tensor.value());
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return Value((Tensor<int32_t>)tensor.value());
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return Value((Tensor<int8_t>)tensor.value());
  } else {
    assert(false);
  }
}

template <typename T>
std::string printTypeWithScalarTensor(std::string type, Tensor<T> tensor) {
  std::stringstream str;
  if (tensor.isScalar()) {
    str << type << "(" << tensor.values[0] << ")";
  } else {
    str << type << "[](";
    for (auto v : tensor.values) {
      str << v << ",";
    }
    str << ")";
  }
  return str.str();
}

std::string Value::toString() const {
  if (auto tensor = getTensor<int8_t>(); tensor) {
    return printTypeWithScalarTensor("int8_t", *tensor);
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return printTypeWithScalarTensor("int16_t", *tensor);
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return printTypeWithScalarTensor("int32_t", *tensor);
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return printTypeWithScalarTensor("int64_t", *tensor);
  } else if (auto tensor = getTensor<uint8_t>(); tensor) {
    return printTypeWithScalarTensor("uint8_t", *tensor);
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return printTypeWithScalarTensor("uint16_t", *tensor);
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return printTypeWithScalarTensor("uint32_t", *tensor);
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return printTypeWithScalarTensor("uint64_t", *tensor);
  } else {
    assert(false);
  }
}

size_t getCorrespondingPrecision(size_t originalPrecision) {
  if (originalPrecision <= 8) {
    return 8;
  }
  if (originalPrecision <= 16) {
    return 16;
  }
  if (originalPrecision <= 32) {
    return 32;
  }
  if (originalPrecision <= 64) {
    return 64;
  }
  assert(false);
}

} // namespace values
} // namespace concretelang
