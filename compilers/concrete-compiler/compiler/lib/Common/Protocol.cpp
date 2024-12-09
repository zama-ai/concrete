// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Protocol.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "llvm/ADT/Hashing.h"
#include <memory>
#include <stdlib.h>

namespace concretelang {
namespace protocol {

/// Helper function turning a protocol `Shape` object into a vector of
/// dimensions.
std::vector<size_t>
protoShapeToDimensions(const Message<concreteprotocol::Shape> &shape) {
  return protoShapeToDimensions(shape.asReader());
}

std::vector<size_t>
protoShapeToDimensions(concreteprotocol::Shape::Reader reader) {
  auto output = std::vector<size_t>();
  for (auto dim : reader.getDimensions()) {
    output.push_back(dim);
  }
  return output;
}

/// Helper function turning a protocol `Shape` object into a vector of
/// dimensions.
Message<concreteprotocol::Shape>
dimensionsToProtoShape(const std::vector<size_t> &input) {
  auto output = Message<concreteprotocol::Shape>();
  auto dimensions = output.asBuilder().initDimensions(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    dimensions.set(i, input[i]);
  }
  return output;
}

template <typename Message> size_t hashMessage(Message &mess) {
  return llvm::hash_value(MessageToJSONString(mess));
}

} // namespace protocol
} // namespace concretelang
