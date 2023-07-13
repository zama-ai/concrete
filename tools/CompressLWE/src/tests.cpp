#include "compress_lwe/defines.h"
#include "compress_lwe/library.h"
#include "compress_lwe/serialize.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <libhcs++.hpp>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>

using namespace std::chrono;
using namespace comp;

mpz_class *to_raw2(const mpz &a) { return (mpz_class *)a.ptr; }

void test_single_lwe_compress() {

  uint64_t n = 100;
  uint64_t log_q = 64;
  bool binaryKeys = true;
  size_t s = 1;

  // Generate a random vector<uint64_t> of size n with values in [0, q) and call
  // it the secret key
  std::vector<uint64_t> lwe_key(n);
  for (uint64_t i = 0; i < n; i++) {
    if (binaryKeys) {
      lwe_key[i] = sample(1);
    } else {
      lwe_key[i] = sample(log_q);
    }
  }

  // Generate a ciphertext
  std::vector<uint64_t> lwe_ct(n + 1);
  for (uint64_t i = 0; i < n + 1; i++) {
    lwe_ct[i] = sample(log_q);
  }

  // Call the generate Keys function with the lwe_key
  comp::FullKeys key = generateKeys(lwe_key, s);

  comp::CompressionKey key2 = key.compression_key();

  // Call the compressSingle function with the compression key and the
  // ciphertext
  comp::mpz compressed_ct = compressSingle(key2, lwe_ct.data(), n);

  // Call the decryptCompressedSingle function with the compressed
  // ciphertext and the paillier private key
  uint64_t decrypted = decryptCompressedSingle(compressed_ct, *key.ahe_sk);

  uint64_t lwe_decrypted = decryptLWE(lwe_ct.data(), lwe_key);
  printf("%lu <--> %lu\n", lwe_decrypted, decrypted);

  assert(decrypted == lwe_decrypted);

  // print success message
  std::cout << "Single Test Passed!" << std::endl;
}

void test_batched_compress(uint64_t num_cts, uint64_t s) {
  uint64_t n = 10;
  uint64_t log_q = 64;
  uint64_t p = 32;
  bool binaryKeys = true;

  assert(log_q <= 64);

  // Get starting timepoint
  system_clock::time_point start = high_resolution_clock::now();
  // Generate a lwe key of size n with values in [0, q)
  std::vector<uint64_t> lwe_key(n);
  for (uint64_t i = 0; i < n; i++) {
    if (binaryKeys) {
      lwe_key[i] = sample(1);
    } else {
      lwe_key[i] = sample(log_q);
    }
  }
  // Get ending timepoint
  system_clock::time_point stop = high_resolution_clock::now();
  // Get duration. Substart timepoints to
  microseconds d_lwe_key = duration_cast<microseconds>(stop - start);

  start = stop;

  // Generate a batch of lwe ciphertexts
  std::vector<uint64_t> lwe_cts(num_cts * (n + 1));
  for (uint64_t i = 0; i < num_cts * (n + 1); i++) {
    lwe_cts[i] = sample(log_q);
  }
  // Get ending timepoint
  stop = high_resolution_clock::now();
  // Get duration. Substart timepoints to
  microseconds d_lwe_cts = duration_cast<microseconds>(stop - start);

  start = stop;

  comp::FullKeys key = generateKeys(lwe_key, s);

  // Get ending timepoint
  stop = high_resolution_clock::now();
  // Get duration. Substart timepoints to
  microseconds d_keygen = duration_cast<microseconds>(stop - start);

  comp::CompressionKey key2 = key.compression_key();

  start = stop;

  comp::CompressedCiphertext compressed_ct =
      compressBatched(key2, lwe_cts.data(), n, num_cts);

  // Get ending timepoint
  stop = high_resolution_clock::now();
  // Get duration. Substart timepoints to
  microseconds d_compress = duration_cast<microseconds>(stop - start);

  start = stop;

  std::vector<uint64_t> decrypted =
      decryptCompressedBatched(compressed_ct, *key.ahe_sk, num_cts);

  // Get ending timepoint
  stop = high_resolution_clock::now();
  // Get duration. Substart timepoints to
  microseconds d_decrypt = duration_cast<microseconds>(stop - start);

  bool test = true;
  for (uint64_t i = 0; i < num_cts; i++) {
    uint64_t lwe_decrypted = decryptLWE(lwe_cts.data() + i * (n + 1), lwe_key);

    // printf("%lu <--> %lu\n", lwe_decrypted, decrypted[i]);
    if (decrypted[i] != lwe_decrypted) {
      test = false;
    }
  }
  assert(test);
  if (test) {
    std::cout << "Encrypt-first Test Passed!" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
              << std::endl;
    std::cout << "S: " << s << " ; Ciphers:  " << num_cts << std::endl;
    std::cout << "Number of Damgard CTs:  " << compressed_ct.ahe_cts.size()
              << " ciphers" << std::endl;
    std::cout << "Time taken by compress: " << d_compress.count() / 1000
              << " ms" << std::endl;

    std::cout << "Time taken by LWE key:  " << d_lwe_key.count()
              << " microseconds" << std::endl;
    std::cout << "Time taken by LWE cts:  " << d_lwe_cts.count()
              << " microseconds" << std::endl;
    std::cout << "Time taken by keygen:   " << d_keygen.count()
              << " microseconds" << std::endl;
    std::cout << "Time taken by decrypt:  " << d_decrypt.count()
              << " microseconds" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
              << std::endl;

  } else {
    std::cout << "Test Failed!" << std::endl;
  }
}

void test_batched_serialize(uint64_t num_cts, uint64_t s) {

  uint64_t n = 100;
  uint64_t log_q = 64;
  uint64_t p = 32;
  bool binaryKeys = true;

  assert(log_q <= 64);

  // Generate a lwe key of size n with values in [0, q)
  std::vector<uint64_t> lwe_key(n);
  for (uint64_t i = 0; i < n; i++) {
    if (binaryKeys) {
      lwe_key[i] = sample(1);
    } else {
      lwe_key[i] = sample(log_q);
    }
  }

  // Generate a batch of lwe ciphertexts
  std::vector<uint64_t> lwe_cts(num_cts * (n + 1));
  for (uint64_t i = 0; i < num_cts * (n + 1); i++) {
    lwe_cts[i] = sample(log_q);
  }

  comp::FullKeys key_ = generateKeys(lwe_key, s);

  std::stringstream stream1;

  comp::writeFullKeys(stream1, key_);
  stream1.seekg(0, std::ios::beg);

  comp::FullKeys key;
  comp::readFullKeys(stream1, key);

  std::stringstream stream2;

  comp::CompressionKey key2_ = key_.compression_key();

  comp::writeCompKeys(stream2, key2_);
  stream2.seekg(0, std::ios::beg);

  comp::CompressionKey key2;

  comp::readCompKeys(stream2, key2);

  comp::CompressedCiphertext compressed_ct_ =
      compressBatchedEncryptFirst(key2, lwe_cts.data(), n, num_cts);

  std::stringstream stream3;

  comp::writeCompCt(stream3, compressed_ct_);

  comp::CompressedCiphertext compressed_ct;

  stream3.seekg(0, std::ios::beg);

  comp::readCompCt(stream3, compressed_ct);

  std::vector<uint64_t> decrypted =
      decryptCompressedBatched(compressed_ct, *key.ahe_sk, num_cts);

  bool test = true;
  for (uint64_t i = 0; i < num_cts; i++) {
    uint64_t lwe_decrypted = decryptLWE(lwe_cts.data() + i * (n + 1), lwe_key);

    // printf("%lu <--> %lu\n", lwe_decrypted, decrypted[i]);
    if (decrypted[i] != lwe_decrypted) {
      test = false;
    }
  }
  if (test) {
    std::cout << "Serialize Test Passed!" << std::endl;
  } else {
    std::cout << "Test Failed!" << std::endl;
  }
}
int main() {
  srand(time(nullptr));
  test_single_lwe_compress();
  uint64_t num_cts = 784;
  for (size_t s = 1; s < 8; ++s) {
    test_batched_compress(num_cts, s);
  }

  test_batched_serialize(20, 1);

  return 0;
}
