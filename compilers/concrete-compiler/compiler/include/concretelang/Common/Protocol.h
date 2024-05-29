// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_PROTOCOL_H
#define CONCRETELANG_COMMON_PROTOCOL_H

#include "boost/outcome.h"
#include "capnp/common.h"
#include "capnp/compat/json.h"
#include "capnp/message.h"
#include "capnp/serialize-packed.h"
#include "capnp/serialize.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "kj/common.h"
#include "kj/exception.h"
#include "kj/io.h"
#include "kj/std/iostream.h"
#include "kj/string.h"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

using concretelang::error::Result;
using concretelang::error::StringError;

const uint64_t MAX_SEGMENT_SIZE = capnp::MAX_SEGMENT_WORDS;

namespace concretelang {
namespace protocol {

/// Arena carrying capnp messages.
///
/// This type packs a message with an arena used to store the data in a single
/// object.
///
/// Rationale:
/// ----------
///
/// Capnproto is a performance-oriented serialization framework, which
/// approaches the problem by constructing a memory representation that is
/// already equivalent to the serialized binary representation.
///
/// To make that possible and as fast as possible, they use an arena-passing
/// programming model, which makes serialization fast, but is also pretty
/// invasive:
/// + The top-level message being constructed must be known in advanced, so as
///   to properly initialize the MallocMessageBuilder.
/// + The parent message builder must be passed to a function creating a child
///   message, so as to properly initialize the field, and fill it.
/// + The arena must be managed at the top level to ensure that the messages are
///   always pointing to valid memory locations.
///
/// In the compiler, we use the concrete-protocol messages for slightly more
/// than just serialization, and having a less-than-optimal serialization speed
/// is not a big problem for now. For this reason, it makes sense to make the
/// use of the capnp messages a little more ergonomic, which is what this type
/// allows.
template <typename MessageType> struct Message {

  Message() : message(nullptr) {
    regionBuilder = new capnp::MallocMessageBuilder();
    message = regionBuilder->initRoot<MessageType>();
  }

  Message(const typename MessageType::Reader &reader) : message(nullptr) {
    regionBuilder = new capnp::MallocMessageBuilder(
        std::min(reader.totalSize().wordCount, MAX_SEGMENT_SIZE),
        capnp::AllocationStrategy::FIXED_SIZE);
    regionBuilder->setRoot(reader);
    message = regionBuilder->getRoot<MessageType>();
  }

  Message(const Message &input) : message(nullptr) {
    regionBuilder = new capnp::MallocMessageBuilder(
        std::min(input.message.asReader().totalSize().wordCount,
                 MAX_SEGMENT_SIZE),
        capnp::AllocationStrategy::FIXED_SIZE);
    regionBuilder->setRoot(input.message.asReader());
    message = regionBuilder->getRoot<MessageType>();
  }

  Message &operator=(const typename MessageType::Reader &reader) {
    if (regionBuilder) {
      delete regionBuilder;
    }
    regionBuilder = new capnp::MallocMessageBuilder(
        std::min(reader.totalSize().wordCount, MAX_SEGMENT_SIZE),
        capnp::AllocationStrategy::FIXED_SIZE);
    regionBuilder->setRoot(reader);
    message = regionBuilder->getRoot<MessageType>();
    return *this;
  }

  Message &operator=(const Message &input) {
    if (this != &input) {
      if (regionBuilder) {
        delete regionBuilder;
      }
      regionBuilder = new capnp::MallocMessageBuilder(
          std::min(input.message.asReader().totalSize().wordCount,
                   MAX_SEGMENT_SIZE),
          capnp::AllocationStrategy::FIXED_SIZE);
      regionBuilder->setRoot(input.message.asReader());
      message = regionBuilder->getRoot<MessageType>();
    }

    return *this;
  }

  Message(Message &&input) : message(nullptr) {
    regionBuilder = input.regionBuilder;
    message = input.message;
    input.regionBuilder = nullptr;
  }

  Message &operator=(Message &&input) {
    if (this != &input) {
      if (regionBuilder) {
        delete regionBuilder;
      }
      regionBuilder = input.regionBuilder;
      message = input.message;
      input.regionBuilder = nullptr;
    }
    return *this;
  }

  ~Message() {
    if (regionBuilder) {
      delete regionBuilder;
    }
  }

  typename MessageType::Reader asReader() const { return message.asReader(); }

  typename MessageType::Builder asBuilder() { return message; }

  Result<void> writeBinaryToFd(int fd) const {
    try {
      capnp::writeMessageToFd(fd, *regionBuilder);
      return outcome::success();
    } catch (const kj::Exception &e) {
      return StringError("Failed to write message to file descriptor: ")
             << e.getDescription().cStr();
    } catch (...) {
      return StringError("Failed to write message to file descriptor.");
    }
  }

  Result<void> writeBinaryToOstream(std::ostream &ostream) const {
    try {
      kj::std::StdOutputStream kjOstream(ostream);
      capnp::writeMessage(kjOstream, *regionBuilder);
    } catch (const kj::Exception &e) {
      return StringError("Failed to write message to ostream: ")
             << e.getDescription().cStr();
    } catch (...) {
      return StringError("Failed to write message to ostream.");
    }
    ostream.flush();
    if (!ostream.good()) {
      return StringError(
          "Failed to write message to ostream. Ended up in bad state.");
    }
    return outcome::success();
  }

  Result<std::string> writeBinaryToString() const {
    auto ostream = std::ostringstream();
    OUTCOME_TRYV(this->writeBinaryToOstream(ostream));
    return outcome::success(ostream.str());
  }

  Result<std::string> writeJsonToString() const {
    try {
      capnp::JsonCodec json;
      kj::String output = json.encode(this->message.asReader());
      return outcome::success(std::string(output.cStr(), output.size()));
    } catch (const kj::Exception &e) {
      return outcome::failure(
          StringError("Failed to write message to json string: ")
          << e.getDescription().cStr());
    } catch (...) {
      return outcome::failure(
          StringError("Failed to write message to json string."));
    }
  }

  Result<void> readBinaryFromFd(int fd) {
    try {
      capnp::readMessageCopyFromFd(fd, *regionBuilder);
      this->message = regionBuilder->getRoot<MessageType>();
      return outcome::success();
    } catch (const kj::Exception &e) {
      return StringError("Failed to read message from file descriptor: ")
             << e.getDescription().cStr();
    } catch (...) {
      return StringError("Failed to read message from file descriptor.");
    }
  }

  Result<void>
  readBinaryFromIstream(std::istream &istream,
                        capnp::ReaderOptions options = capnp::ReaderOptions()) {
    try {
      kj::std::StdInputStream kjIstream(istream);
      capnp::readMessageCopy(kjIstream, *regionBuilder, options);
      this->message = regionBuilder->getRoot<MessageType>();
      return outcome::success();
    } catch (const kj::Exception &e) {
      return StringError("Failed to read message from istream: ")
             << e.getDescription().cStr();
    } catch (...) {
      return StringError("Failed to read message from istream.");
    }
  }

  Result<void>
  readBinaryFromString(const std::string &input,
                       capnp::ReaderOptions options = capnp::ReaderOptions()) {
    auto istream = std::istringstream(input);
    return this->readBinaryFromIstream(istream, options);
  }

  Result<void> readJsonFromString(const std::string &input) {
    try {
      capnp::JsonCodec json;
      kj::StringPtr stringPointer(input.c_str(), input.size());
      this->message = this->regionBuilder->template initRoot<MessageType>();
      json.decode(stringPointer, this->message);
      return outcome::success();
    } catch (const kj::Exception &e) {
      return StringError("Failed to read message from json string: ")
             << e.getDescription().cStr();
    } catch (...) {
      return StringError("Failed to read message from json string.");
    }
  }

  std::string debugString() const { return writeJsonToString().value(); }

private:
  capnp::MallocMessageBuilder *regionBuilder;
  typename MessageType::Builder message;
};

template struct Message<concreteprotocol::ProgramInfo>;
template struct Message<concreteprotocol::CircuitEncodingInfo>;
template struct Message<concreteprotocol::ProgramEncodingInfo>;
template struct Message<concreteprotocol::Value>;
template struct Message<concreteprotocol::GateInfo>;

/// Helper function turning a vector of integers to a payload.
template <typename T>
Message<concreteprotocol::Payload>
vectorToProtoPayload(const std::vector<T> &input) {
  auto output = Message<concreteprotocol::Payload>();
  auto elmsPerBlob = capnp::MAX_TEXT_SIZE / sizeof(T);
  auto remainingElms = input.size() % elmsPerBlob;
  auto nbBlobs = (input.size() / elmsPerBlob) + (remainingElms > 0);
  auto dataBuilder = output.asBuilder().initData(nbBlobs);
  // Process all but the last blob, which store as much as `Data` allow.
  if (nbBlobs > 1) {
    for (size_t blobIndex = 0; blobIndex < nbBlobs - 1; blobIndex++) {
      auto blobPtr = input.data() + blobIndex * elmsPerBlob;
      auto blobLen = elmsPerBlob * sizeof(T);
      dataBuilder.set(
          blobIndex,
          capnp::Data::Reader(reinterpret_cast<const unsigned char *>(blobPtr),
                              blobLen));
    }
  }

  // Process the last blob which store the remainder.
  if (nbBlobs > 0) {
    auto lastBlobIndex = nbBlobs - 1;
    auto lastBlobPtr = input.data() + lastBlobIndex * elmsPerBlob;
    auto lastBlobLen = remainingElms * sizeof(T);
    dataBuilder.set(
        lastBlobIndex,
        capnp::Data::Reader(
            reinterpret_cast<const unsigned char *>(lastBlobPtr), lastBlobLen));
  }

  return output;
}

/// Helper function turning a payload to a vector of integers.
template <typename T>
std::vector<T>
protoPayloadToVector(const Message<concreteprotocol::Payload> &input) {
  auto payloadData = input.asReader().getData();
  auto elmsPerBlob = capnp::MAX_TEXT_SIZE / sizeof(T);
  size_t totalPayloadSize = 0;
  for (auto blob : payloadData) {
    totalPayloadSize += blob.size();
  }
  assert(totalPayloadSize % sizeof(T) == 0);
  auto dataSize = totalPayloadSize / sizeof(T);
  auto output = std::vector<T>();
  output.resize(dataSize);
  for (size_t blobIndex = 0; blobIndex < payloadData.size(); blobIndex++) {
    auto blobData = payloadData[blobIndex];
    auto blobPtr = output.data() + blobIndex * elmsPerBlob;
    std::memcpy(blobPtr, blobData.begin(), blobData.size());
  }
  return output;
}

/// Helper function turning a payload to a shared vector of integers on the
/// heap.
template <typename T>
std::shared_ptr<std::vector<T>>
protoPayloadToSharedVector(const Message<concreteprotocol::Payload> &input) {
  auto payloadData = input.asReader().getData();
  size_t elmsPerBlob = capnp::MAX_TEXT_SIZE / sizeof(T);
  size_t totalPayloadSize = 0;
  for (auto blob : payloadData) {
    totalPayloadSize += blob.size();
  }
  assert(totalPayloadSize % sizeof(T) == 0);
  size_t dataSize = totalPayloadSize / sizeof(T);
  auto output = std::make_shared<std::vector<T>>();
  output->resize(dataSize);
  for (size_t blobIndex = 0; blobIndex < payloadData.size(); blobIndex++) {
    auto blobData = payloadData[blobIndex];
    auto blobPtr = output->data() + blobIndex * elmsPerBlob;
    std::memcpy(blobPtr, blobData.begin(), blobData.size());
  }
  return output;
}

/// Helper function turning a protocol `Shape` object into a vector of
/// dimensions.
std::vector<size_t>
protoShapeToDimensions(const Message<concreteprotocol::Shape> &shape);

/// Helper function turning a protocol `Shape` object into a vector of
/// dimensions.
Message<concreteprotocol::Shape>
dimensionsToProtoShape(const std::vector<size_t> &input);

template <typename MessageType> size_t hashMessage(Message<MessageType> &mess);

} // namespace protocol
} // namespace concretelang

#endif
