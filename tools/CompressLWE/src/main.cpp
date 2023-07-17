#include <iostream>
#include <cassert>
#include "library.h"

using namespace std;

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
        if (binaryKeys){
            lwe_decryption_key[i] = sample(1);
        } else {
            lwe_decryption_key[i] = sample(log_q);
        }
    }

    // Generate a batch of lwe ciphertexts
    uint64_t num_cts = 10;
    std::vector<std::vector<uint64_t>> lwe_cts(num_cts);
    for (int i = 0; i < num_cts; i++) {
        lwe_cts[i] = std::vector<uint64_t>(n + 1);
        for (int j = 0; j < n + 1; j++) {
            lwe_cts[i][j] = sample(log_q);
        }
    }

    // 1- Cast LWE params and Generate Paillier and Compression keys
    // s is a parameter which controls how compressed the ciphertexts are
    // The higher s is, the more compression is achieved
    // However, the higher s is, the more time it takes to compress 
    LWEParams params(n, log_q, p, binaryKeys);
    Keys* keys = generateKeys(lwe_decryption_key, s);

    // 2- Compress a batch of LWE ciphertexts into 1 CompressedCiphertext [which is a vector of additive ciphers under the hood]
    CompressedCiphertext compressed_ct = compressBatched(
            keys->compKey,
            keys->ahe_pk,
            lwe_cts,
            params
    );

    // 3- Decrypt the batch of compressed ciphertexts
    mpz_vec decrypted = decryptCompressedBatched(
            compressed_ct,
            keys->ahe_sk,
            params,
            num_cts,
            true
    );

    // Make sure decryption works correctly
    for (uint64_t i = 0; i < num_cts; i++) {
        mpz_class lwe_decrypted = decryptLWE(lwe_cts[i], lwe_decryption_key, params);
        assert(decrypted[i] == lwe_decrypted);
    }
    std::cout << "Batched Test Passed!" << std::endl;

    return 0;
}