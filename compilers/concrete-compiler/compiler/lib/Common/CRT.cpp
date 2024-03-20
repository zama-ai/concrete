// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <cstddef>
#include <stdio.h>

#include "concretelang/Common/CRT.h"

namespace concretelang {
namespace crt {
uint64_t productOfModuli(std::vector<int64_t> moduli) {
  uint64_t product = 1;
  for (auto modulus : moduli) {
    product *= modulus;
  }
  return product;
}

std::vector<int64_t> crt(std::vector<int64_t> moduli, uint64_t val) {
  std::vector<int64_t> remainders(moduli.size(), 0);

  for (size_t i = 0; i < moduli.size(); i++) {
    remainders[i] = val % moduli[i];
  }
  return remainders;
}

// https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/
// Returns modulo inverse of a with respect
// to m using extended Euclid Algorithm
// Assumption: a and m are coprimes, i.e.,
// gcd(a, m) = 1
int64_t modInverse(int64_t a, int64_t m) {
  int64_t m0 = m;
  int64_t y = 0, x = 1;

  if (m == 1)
    return 0;

  while (a > 1) {
    // q is quotient
    int64_t q = a / m;
    int64_t t = m;

    // m is remainder now, process same as
    // Euclid's algo
    m = a % m;
    a = t;
    t = y;

    // Update y and x
    y = x - q * y;
    x = t;
  }

  // Make x positive
  if (x < 0)
    x += m0;

  return x;
}

uint64_t iCrt(std::vector<int64_t> moduli, std::vector<int64_t> remainders) {
  // Compute the product of moduli
  int64_t product = productOfModuli(moduli);

  int64_t result = 0;

  // Apply above formula
  for (size_t i = 0; i < remainders.size(); i++) {
    int tmp = product / moduli[i];
    result += remainders[i] * modInverse(tmp, moduli[i]) * tmp;
  }

  return result % product;
}

uint64_t encode(int64_t plaintext, uint64_t modulus, uint64_t product) {
  // values are represented on the interval [0; product[ so we represent
  // plaintext on this interval
  if (plaintext < 0) {
    plaintext = product + plaintext;
  }
  __uint128_t m = plaintext % modulus;
  return m * ((__uint128_t)(1) << 64) / modulus;
}

uint64_t decode(uint64_t val, uint64_t modulus) {
  auto result = (__uint128_t)val * (__uint128_t)modulus;
  result = result + ((result & ((__uint128_t)(1) << 63)) << 1);
  result = result / ((__uint128_t)(1) << 64);
  return (uint64_t)result % modulus;
}
} // namespace crt
} // namespace concretelang
