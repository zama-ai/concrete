// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

// generated: see genDynamicRandCall.py

#include <cassert>
#include <vector>

#include "concretelang/ClientLib/Types.h"
#include "concretelang/ServerLib/DynamicArityCall.h"
#include "concretelang/ServerLib/ServerLambda.h"

namespace concretelang {
namespace serverlib {

TensorData multi_arity_call_dynamic_rank(void *(*func)(void *...),
                                         std::vector<void *> args,
                                         size_t rank) {
  using concretelang::clientlib::MemRefDescriptor;
  constexpr auto convert = concretelang::clientlib::tensorDataFromMemRef;
  switch (rank) {
  case 0: {
    auto m = multi_arity_call((MemRefDescriptor<1>(*)(void *...))func, args);
    return convert(1, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 1: {
    auto m = multi_arity_call((MemRefDescriptor<2>(*)(void *...))func, args);
    return convert(2, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 2: {
    auto m = multi_arity_call((MemRefDescriptor<3>(*)(void *...))func, args);
    return convert(3, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 3: {
    auto m = multi_arity_call((MemRefDescriptor<4>(*)(void *...))func, args);
    return convert(4, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 4: {
    auto m = multi_arity_call((MemRefDescriptor<5>(*)(void *...))func, args);
    return convert(5, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 5: {
    auto m = multi_arity_call((MemRefDescriptor<6>(*)(void *...))func, args);
    return convert(6, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 6: {
    auto m = multi_arity_call((MemRefDescriptor<7>(*)(void *...))func, args);
    return convert(7, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 7: {
    auto m = multi_arity_call((MemRefDescriptor<8>(*)(void *...))func, args);
    return convert(8, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 8: {
    auto m = multi_arity_call((MemRefDescriptor<9>(*)(void *...))func, args);
    return convert(9, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 9: {
    auto m = multi_arity_call((MemRefDescriptor<10>(*)(void *...))func, args);
    return convert(10, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 10: {
    auto m = multi_arity_call((MemRefDescriptor<11>(*)(void *...))func, args);
    return convert(11, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 11: {
    auto m = multi_arity_call((MemRefDescriptor<12>(*)(void *...))func, args);
    return convert(12, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 12: {
    auto m = multi_arity_call((MemRefDescriptor<13>(*)(void *...))func, args);
    return convert(13, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 13: {
    auto m = multi_arity_call((MemRefDescriptor<14>(*)(void *...))func, args);
    return convert(14, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 14: {
    auto m = multi_arity_call((MemRefDescriptor<15>(*)(void *...))func, args);
    return convert(15, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 15: {
    auto m = multi_arity_call((MemRefDescriptor<16>(*)(void *...))func, args);
    return convert(16, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 16: {
    auto m = multi_arity_call((MemRefDescriptor<17>(*)(void *...))func, args);
    return convert(17, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 17: {
    auto m = multi_arity_call((MemRefDescriptor<18>(*)(void *...))func, args);
    return convert(18, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 18: {
    auto m = multi_arity_call((MemRefDescriptor<19>(*)(void *...))func, args);
    return convert(19, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 19: {
    auto m = multi_arity_call((MemRefDescriptor<20>(*)(void *...))func, args);
    return convert(20, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 20: {
    auto m = multi_arity_call((MemRefDescriptor<21>(*)(void *...))func, args);
    return convert(21, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 21: {
    auto m = multi_arity_call((MemRefDescriptor<22>(*)(void *...))func, args);
    return convert(22, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 22: {
    auto m = multi_arity_call((MemRefDescriptor<23>(*)(void *...))func, args);
    return convert(23, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 23: {
    auto m = multi_arity_call((MemRefDescriptor<24>(*)(void *...))func, args);
    return convert(24, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 24: {
    auto m = multi_arity_call((MemRefDescriptor<25>(*)(void *...))func, args);
    return convert(25, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 25: {
    auto m = multi_arity_call((MemRefDescriptor<26>(*)(void *...))func, args);
    return convert(26, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 26: {
    auto m = multi_arity_call((MemRefDescriptor<27>(*)(void *...))func, args);
    return convert(27, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 27: {
    auto m = multi_arity_call((MemRefDescriptor<28>(*)(void *...))func, args);
    return convert(28, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 28: {
    auto m = multi_arity_call((MemRefDescriptor<29>(*)(void *...))func, args);
    return convert(29, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 29: {
    auto m = multi_arity_call((MemRefDescriptor<30>(*)(void *...))func, args);
    return convert(30, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 30: {
    auto m = multi_arity_call((MemRefDescriptor<31>(*)(void *...))func, args);
    return convert(31, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 31: {
    auto m = multi_arity_call((MemRefDescriptor<32>(*)(void *...))func, args);
    return convert(32, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }
  case 32: {
    auto m = multi_arity_call((MemRefDescriptor<33>(*)(void *...))func, args);
    return convert(33, m.allocated, m.aligned, m.offset, m.sizes, m.strides);
  }

  default:
    assert(false);
  }
}

} // namespace serverlib
} // namespace concretelang
