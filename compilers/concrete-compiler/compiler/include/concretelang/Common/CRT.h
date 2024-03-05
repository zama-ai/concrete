// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_CRT_H_
#define CONCRETELANG_COMMON_CRT_H_

#include <cstdint>
#include <vector>

namespace concretelang {
namespace crt {

/// Compute the product of the moduli of the crt decomposition.
///
/// \param moduli The moduli of the crt decomposition
/// \returns The product of moduli
uint64_t productOfModuli(std::vector<int64_t> moduli);

/// Compute the crt decomposition of a `val` according the given `moduli`.
///
/// \param moduli The moduli to compute the decomposition.
/// \param val The value to decompose.
/// \returns The remainders.
std::vector<int64_t> crt(std::vector<int64_t> moduli, uint64_t val);

/// Compute the inverse of the crt decomposition.
///
/// \param moduli The moduli used to compute the inverse decomposition.
/// \param remainders The remainders of the decomposition.
uint64_t iCrt(std::vector<int64_t> moduli, std::vector<int64_t> remainders);

/// Encode the plaintext with the given modulus and the product of moduli of the
/// crt decomposition
uint64_t encode(int64_t plaintext, uint64_t modulus, uint64_t product);

/// Decode follow the crt encoding
uint64_t decode(uint64_t val, uint64_t modulus);

} // namespace crt
} // namespace concretelang

#endif
