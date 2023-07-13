#include "compress_lwe/library.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <libhcs++.hpp>
#include <vector>

using namespace std;
using namespace comp;

int main() {
  // LWE parameters
  uint64_t n = 630;
  uint64_t log_q = 64;
  uint64_t p = 32;
  bool binaryKeys = true;

  // This parameter controls how much comrpesssion is achieved
  size_t s = 1;

  assert(log_q <= 64);

  // Generate lwe decryption keys of size n with values in [0, q)
  std::vector<uint64_t> lwe_decryption_key(n);
  for (int i = 0; i < n; i++) {
    if (binaryKeys) {
      lwe_decryption_key[i] = sample(1);
    } else {
      lwe_decryption_key[i] = sample(log_q);
    }
  }

  // Generate a batch of lwe ciphertexts
  uint64_t num_cts = 10;
  std::vector<uint64_t> lwe_cts(num_cts * (n + 1));
  for (int i = 0; i < num_cts * (n + 1); i++) {
    lwe_cts[i] = sample(log_q);
  }

  // 1- Cast LWE params and Generate Paillier and Compression keys
  // s is a parameter which controls how compressed the ciphertexts are
  // The higher s is, the more compression is achieved
  // However, the higher s is, the more time it takes to compress

  FullKeys keys = generateKeys(lwe_decryption_key, s);

  CompressionKey key = keys.compression_key();

  // 2- Compress a batch of LWE ciphertexts into 1 CompressedCiphertext [which
  // is a vector of additive ciphers under the hood]
  CompressedCiphertext compressed_ct =
      compressBatched(key, lwe_cts.data(), n, num_cts);

  // 3- Decrypt the batch of compressed ciphertexts
  std::vector<uint64_t> decrypted =
      decryptCompressedBatched(compressed_ct, *keys.ahe_sk, num_cts);

  // Make sure decryption works correctly
  for (uint64_t i = 0; i < num_cts; i++) {
    uint64_t lwe_decrypted =
        decryptLWE(lwe_cts.data() + (i * (n + 1)), lwe_decryption_key);

    std::cout << decrypted[i] << " = " << lwe_decrypted << std::endl;
    if (decrypted[i] != lwe_decrypted) {
      exit(-1);
    }
  }
  std::cout << "Batched Test Passed!" << std::endl;

  return 0;
}
