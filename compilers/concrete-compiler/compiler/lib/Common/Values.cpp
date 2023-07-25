// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Values.h"
#include <cstddef>
#include <cstdint>
#include <stdlib.h>

using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::protocol::dimensionsToProtoShape;
using concretelang::protocol::protoDataToVector;
using concretelang::protocol::protoShapeToDimensions;
using concretelang::protocol::vectorToProtoData;

namespace concretelang {
namespace values {

Value Value::fromRawTransportValue(const TransportValue &transportVal) {
  Value output;
  auto integerPrecision = transportVal.rawinfo().integerprecision();
  auto isSigned = transportVal.rawinfo().issigned();
  auto dimensions = protoShapeToDimensions(transportVal.rawinfo().shape());
  auto data = transportVal.data();
  if (integerPrecision == 8 && isSigned) {
    auto values = protoDataToVector<int8_t>(data);
    output.inner = Tensor<int8_t>{values, dimensions};
  } else if (integerPrecision == 16 && isSigned) {
    auto values = protoDataToVector<int16_t>(data);
    output.inner = Tensor<int16_t>{values, dimensions};
  } else if (integerPrecision == 32 && isSigned) {
    auto values = protoDataToVector<int32_t>(data);
    output.inner = Tensor<int32_t>{values, dimensions};
  } else if (integerPrecision == 64 && isSigned) {
    auto values = protoDataToVector<int64_t>(data);
    output.inner = Tensor<int64_t>{values, dimensions};
  } else if (integerPrecision == 8 && !isSigned) {
    auto values = protoDataToVector<uint8_t>(data);
    output.inner = Tensor<uint8_t>{values, dimensions};
  } else if (integerPrecision == 16 && !isSigned) {
    auto values = protoDataToVector<uint16_t>(data);
    output.inner = Tensor<uint16_t>{values, dimensions};
  } else if (integerPrecision == 32 && !isSigned) {
    auto values = protoDataToVector<uint32_t>(data);
    output.inner = Tensor<uint32_t>{values, dimensions};
  } else if (integerPrecision == 64 && !isSigned) {
    auto values = protoDataToVector<uint64_t>(data);
    output.inner = Tensor<uint64_t>{values, dimensions};
  } else {
    assert(false);
  }

  return output;
}

TransportValue Value::intoRawTransportValue() {
  auto output = TransportValue();
  auto rawInfo = new concreteprotocol::RawInfo();
  rawInfo->set_allocated_shape(intoProtoShape());
  rawInfo->set_integerprecision(getIntegerPrecision());
  rawInfo->set_issigned(isSigned());
  output.set_allocated_data(intoProtoData());
  output.set_allocated_rawinfo(rawInfo);
  return output;
}

uint32_t Value::getIntegerPrecision() {
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

uint32_t Value::isSigned() {
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

std::string *Value::intoProtoData() {
  if (auto tensor = getTensor<uint8_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<int8_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return vectorToProtoData(tensor.value().values);
  } else {
    assert(false);
  }
}

concreteprotocol::Shape *Value::intoProtoShape() {
  return dimensionsToProtoShape(getDimensions());
}

std::vector<size_t> Value::getDimensions() {
  if (auto tensor = getTensor<uint8_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<uint16_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<uint32_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<uint64_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<int8_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<int16_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<int32_t>(); tensor) {
    return tensor.value().dimensions;
  } else if (auto tensor = getTensor<int64_t>(); tensor) {
    return tensor.value().dimensions;
  } else {
    assert(false);
  }
}

size_t Value::getLength() {
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

bool Value::isCompatibleWithShape(const concreteprotocol::Shape &shape) {
  auto dimensions = getDimensions();
  if ((uint32_t)shape.dimensions_size() != dimensions.size()) {
    return false;
  }
  for (uint32_t i = 0; i < dimensions.size(); i++) {
    if (shape.dimensions(i) != dimensions[i]) {
      return false;
    }
  }
  return true;
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
