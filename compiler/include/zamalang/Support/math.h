// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_SUPPORT_MATH_H_
#define ZAMALANG_SUPPORT_MATH_H_

// Calculates (T)ceil(log2f(v))
// TODO: Replace with some fancy bit twiddling hack
template <typename T> static T ceilLog2(const T v) {
  T tmp = v;
  T log2 = 0;

  while (tmp >>= 1)
    log2++;

  // If more than MSB set, round to next highest power of 2
  if (v & ~((T)1 << log2))
    log2 += 1;

  return log2;
}

#endif
