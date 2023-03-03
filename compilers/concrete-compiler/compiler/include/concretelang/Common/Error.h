// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.
#ifndef CONCRETELANG_COMMON_ERROR_H
#define CONCRETELANG_COMMON_ERROR_H

#include <string>

namespace concretelang {
namespace error {

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

} // namespace error
} // namespace concretelang

#endif
