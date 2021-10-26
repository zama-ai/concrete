#ifndef ZAMALANG_SUPPORT_STRING_ERROR_H
#define ZAMALANG_SUPPORT_STRING_ERROR_H

#include <llvm/Support/Error.h>

namespace mlir {
namespace zamalang {

// Internal error class that allows for composing `llvm::Error`s
// similar to `llvm::createStringError()`, but using stream-like
// composition with `operator<<`.
//
// Example:
//
//   llvm::Error foo(int i, size_t s, ...) {
//      ...
//      if(...) {
//        return StreamStringError()
//               << "Some error message with an integer: "
//               << i << " and a size_t: " << s;
//      }
//      ...
//   }
class StreamStringError {
public:
  StreamStringError(const llvm::StringRef &s) : buffer(s.str()), os(buffer){};
  StreamStringError() : buffer(""), os(buffer){};

  template <typename T> StreamStringError &operator<<(const T &v) {
    this->os << v;
    return *this;
  }

  operator llvm::Error() {
    return llvm::make_error<llvm::StringError>(os.str(),
                                               llvm::inconvertibleErrorCode());
  }

  template <typename T> operator llvm::Expected<T>() {
    return this->operator llvm::Error();
  }

protected:
  std::string buffer;
  llvm::raw_string_ostream os;
};

StreamStringError &operator<<(StreamStringError &se, llvm::Error &err);

} // namespace zamalang
} // namespace mlir

#endif
