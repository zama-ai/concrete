// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_PROTOBUF_H
#define CONCRETELANG_COMMON_PROTOBUF_H

#include <boost/outcome.h>
#include <concretelang/Common/Error.h>
#include <cstddef>
#include <fstream>
#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>
#include <llvm/ADT/Hashing.h>
#include <memory>
#include <stdlib.h>

namespace concretelang {
namespace common {

template <typename Message>
outcome::checked<Message, concretelang::error::StringError>
JSONStringToMessage(std::string content) {
  auto output = Message();
  auto parseRes = google::protobuf::util::JsonStringToMessage(content, &output);
  if (!parseRes.ok()) {
    return concretelang::error::StringError("Failed to parse: ")
           << parseRes.message().as_string() << "\n"
           << content << "\n";
  }

  return output;
}

template <typename Message>
outcome::checked<Message, concretelang::error::StringError>
JSONFileToMessage(std::string path) {
  std::ifstream file(path);
  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));
  if (file.fail()) {
    return concretelang::error::StringError("Cannot read file: ") << path;
  }

  return JSONStringToMessage<Message>(content);
}

template <typename Message>
std::string MessageToJSONString(const Message &mess) {
  std::string output;
  google::protobuf::util::MessageToJsonString(
      (const google::protobuf::Message &)mess, &output);
  return output;
}

template <typename Message> size_t hashMessage(Message &mess) {
  auto outputString = std::string();
  auto res = google::protobuf::util::MessageToJsonString(mess, outputString);
  assert(res.ok());
  auto output = llvm::hash_value(outputString);
  return output;
}

} // namespace common
} // namespace concretelang

#endif
