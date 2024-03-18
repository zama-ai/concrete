// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.
#ifndef CONCRETELANG_COMMON_ERROR_H
#define CONCRETELANG_COMMON_ERROR_H

#include "boost/outcome.h"
#include <string>

namespace concretelang {
namespace error {

/// The type of error used throughout the client/server libs.
class StringError {
public:
  StringError(std::string mesg) : mesg(mesg){};

  std::string mesg;

  StringError &operator<<(const std::string &v) {
    mesg += v;
    return *this;
  }

  StringError &operator<<(const char *v) {
    mesg += std::string(v);
    return *this;
  }

  StringError &operator<<(char *v) {
    mesg += std::string(v);
    return *this;
  }

  template <typename T> inline StringError &operator<<(const T v) {
    mesg += std::to_string(v);
    return *this;
  }
};

/// A result type used throughout the client/server libs.
template <typename T> using Result = outcome::checked<T, StringError>;

} // namespace error
} // namespace concretelang

#endif
