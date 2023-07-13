//
// Created by a2diaa on 4/17/23.
//

#include "compress_lwe/utils.h"
#include <cstdint>
#include <libhcs++.hpp>

namespace comp {

// helper to sample a random uint64_t
uint64_t sample(uint64_t log_q) {
  uint64_t n = ((uint64_t)rand() << 32) ^ ((uint64_t)rand());
  if (log_q < 64)
    return n % (1UL << log_q);
  else
    return n;
}

// uncompressed LWE decryption function, for testing purposes
uint64_t decryptLWE(const uint64_t *lwe_ct, std::vector<uint64_t> lwe_key) {
  uint64_t n = lwe_key.size();

  uint64_t res = lwe_ct[n];
  for (uint64_t i = 0; i < n; i++) {
    res -= lwe_ct[i] * lwe_key[i];
  }
  return res;
}

} // namespace comp
