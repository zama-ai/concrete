// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_MATH_H_
#define CONCRETELANG_SUPPORT_MATH_H_

/// Calculates (T)ceil(log2f(v))
template <typename T> static T ceilLog2(const T v) {
  // TODO: Replace with some fancy bit twiddling hack
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
