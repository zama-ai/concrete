#include <iostream>
#include <cassert>
#include "library.h"

using namespace std;

int main() {
    // LWE parameters
    uint64_t n = 630;
    uint64_t log_q = 64;
    uint64_t p = 32;
    assert(log_q <= 64);

    // Generate lwe decryption keys of size n with values in [0, q)
    std::vector<uint64_t> lwe_decryption_key(n);
    for (int i = 0; i < n; i++) {
        lwe_decryption_key[i] = sample(log_q);
    }

    // Generate a batch of lwe ciphertexts
    uint64_t num_cts = 100;
    std::vector<std::vector<uint64_t>> lwe_cts(num_cts);
    for (int i = 0; i < num_cts; i++) {
        lwe_cts[i] = std::vector<uint64_t>(n + 1);
        for (int j = 0; j < n + 1; j++) {
            lwe_cts[i][j] = sample(log_q);
        }
    }

    // 1- Cast LWE params and Generate Paillier and Compression keys
    LWEParams params(n, log_q, p);
    Keys keys = generateKeys(lwe_decryption_key);

    // 2- Compress a batch of LWE ciphertexts into 1 CompressedCiphertext [a vector of paillier ciphers]
    CompressedCiphertext compressed_ct = compressBatched(
            keys.compKey,
            lwe_cts,
            params
    );

    // 3- Decrypt the batch of compressed ciphertexts
    std::vector<uint64_t> decrypted = decryptCompressedBatched(
            compressed_ct,
            keys.paiKeys.priv_key,
            params,
            num_cts
    );

    // Make sure decryption works correctly
    for (int i = 0; i < num_cts; i++) {
        uint64_t lwe_decrypted = decryptLWE(lwe_cts[i], lwe_decryption_key, params);
        assert(decrypted[i] == lwe_decrypted);
    }
    std::cout << "Batched Test Passed!" << std::endl;

    return 0;
}