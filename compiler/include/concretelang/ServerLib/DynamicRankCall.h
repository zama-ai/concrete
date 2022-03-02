// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SERVERLIB_DYNAMIC_RANK_CALL_H
#define CONCRETELANG_SERVERLIB_DYNAMIC_RANK_CALL_H

#include <vector>

#include "concretelang/ClientLib/Types.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::TensorData;

TensorData multi_arity_call_dynamic_rank(void *(*func)(void *...),
                                         std::vector<void *> args, size_t rank);

} // namespace serverlib
} // namespace concretelang

#endif
