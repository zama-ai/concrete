// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_PROTOCOL_H
#define CONCRETELANG_COMMON_PROTOCOL_H

#include <boost/outcome.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>
#include "concrete-protocol.pb.h"
#include <concretelang/Common/Error.h>

namespace concretelang {
namespace protocol {

/// Helper function turning a vector of integers to a protobuf data binary string.
template <typename T>
std::string *vectorToProtoData(const std::vector<T> &input) {
  auto dataSize = sizeof(T) * input.size();
  auto output = new std::string();
  output->resize(dataSize);
  std::memcpy(output->data(), input.data(), dataSize);
  return output;
}

/// Helper function turning a protobuf data binary string to a vector of integers.
template <typename T>
std::vector<T> protoDataToVector(const std::string &input) {
  assert(input.size() % sizeof(T) == 0);
  auto dataSize = input.size() / sizeof(T);
  auto output = std::vector<T>();
  output.resize(dataSize);
  std::memcpy(output.data(), input.data(), input.size());
  return output;
}

/// Helper function turning a protocol `Shape` object into a vector of dimensions.
std::vector<size_t> protoShapeToDimensions(const concreteprotocol::Shape &shape);

/// Helper function turning a protocol `Shape` object into a vector of dimensions.
concreteprotocol::Shape *dimensionsToProtoShape(const std::vector<size_t> &input);

template <typename Message>
outcome::checked<Message, concretelang::error::StringError>
JSONStringToMessage(std::string content) ;

template <typename Message>
outcome::checked<Message, concretelang::error::StringError>
JSONFileToMessage(std::string path) ;

template <typename Message>
std::string MessageToJSONString(const Message &mess) ;

template <typename Message> size_t hashMessage(Message &mess) ;

} // namespace protocol
} // namespace concretelang

#endif
