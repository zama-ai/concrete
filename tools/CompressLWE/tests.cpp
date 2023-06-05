#include <iostream>
#include <cassert>
#include "library.h"
#include "chrono"
using namespace std::chrono;
void test_single_lwe_compress() {

    uint64_t n = 630;
    uint64_t log_q = 64;
    uint64_t p = 32;

    // Generate a random vector<uint64_t> of size n+1 with values in [0, q) and call it the secret key
    std::vector<uint64_t> lwe_key(n);
    for (int i = 0; i < n; i++) {
        lwe_key[i] = sample(log_q);
    }

    // Generate a ciphertext
    std::vector<uint64_t> lwe_ct(n + 1);
    for (int i = 0; i < n + 1; i++) {
        lwe_ct[i] = sample(log_q);
    }

    // Call the generate Keys function with the lwe_key
    auto key = generateKeys(lwe_key);

    // Call the compressSingle function with the compression key, the ciphertext, and the LWEParams
    LWEParams params(n, log_q, p);
    auto compressed_ct = compressSingle(key.compKey, lwe_ct, params);

    // Call the decryptCompressedSingle function with the compressed ciphertext, the paillier private key, and the LWEParams
    auto decrypted = decryptCompressedSingle(compressed_ct, key.paiKeys.priv_key, params);

    uint64_t lwe_decrypted = decryptLWE(lwe_ct, lwe_key, params);
    assert(decrypted == lwe_decrypted);

    // print success message
    std::cout << "Single Test Passed!" << std::endl;

}

void test_batched_lwe_compress() {
    uint64_t n = 630;
    uint64_t log_q = 64;
    uint64_t p = 32;

    assert(log_q <= 64);

    // Get starting timepoint
    auto start = high_resolution_clock::now();
    // Generate a lwe key of size n with values in [0, q)
    std::vector<uint64_t> lwe_key(n);
    for (int i = 0; i < n; i++) {
        lwe_key[i] = sample(log_q);
    }
    // Get ending timepoint
    auto stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    auto d_lwe_key = duration_cast<microseconds>(stop - start);

    start=stop;

    uint64_t num_cts = 784;
    // Generate a batch of lwe ciphertexts
    std::vector<std::vector<uint64_t>> lwe_cts(num_cts);
    for (int i = 0; i < num_cts; i++) {
        lwe_cts[i] = std::vector<uint64_t>(n + 1);
        for (int j = 0; j < n + 1; j++) {
            lwe_cts[i][j] = sample(log_q);
        }
    }
    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    auto d_lwe_cts = duration_cast<microseconds>(stop - start);

    start = stop;

    auto key = generateKeys(lwe_key);

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    auto d_keygen = duration_cast<microseconds>(stop - start);

    start=stop;

    LWEParams params(n, log_q, p);

    CompressedCiphertext compressed_ct = compressBatched(
            key.compKey,
            lwe_cts,
            params
    );

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    auto d_compress = duration_cast<microseconds>(stop - start);

    start = stop;

    std::vector<uint64_t> decrypted = decryptCompressedBatched(
            compressed_ct,
            key.paiKeys.priv_key,
            params,
            num_cts
    );

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    auto d_decrypt = duration_cast<microseconds>(stop - start);

    for (int i = 0; i < num_cts; i++) {
        uint64_t lwe_decrypted = decryptLWE(lwe_cts[i], lwe_key, params);
        assert(decrypted[i] == lwe_decrypted);
    }
    std::cout << "Batched Test Passed!" << std::endl;
    std::cout << "Time taken by LWE key:  " << d_lwe_key.count() << " microseconds" << std::endl;
    std::cout << "Time taken by LWE cts:  " << d_lwe_cts.count() << " microseconds" << std::endl;
    std::cout << "Time taken by keygen:   " << d_keygen.count() << " microseconds" << std::endl;
    std::cout << "Time taken by compress: " << d_compress.count() << " microseconds" << std::endl;
    std::cout << "Time taken by decrypt:  " << d_decrypt.count() << " microseconds" << std::endl;
}


void test_conversion() {
    for (uint64_t i = 0; i < 10; i++) {
        uint64_t x = sample(64);
        BigNumber x_bn = from64(x);
        uint64_t y = to64(x_bn);
        assert(x == y);
    }
    std::cout << "Conversion Test Passed!" << std::endl;
}


int main() {
    srand(time(nullptr));
    test_conversion();
    test_single_lwe_compress();
    test_batched_lwe_compress();
    return 0;
}