//
// Created by a2diaa on 4/17/23.
//

#include "utils.h"


//helper to sample a random uint64_t
uint64_t sample(uint64_t log_q) {
    uint64_t n = ((uint64_t) rand() << 32) ^ ((uint64_t) rand());
    if (log_q < 64)
        return n % (1UL << log_q);
    else
        return n;
}

// uncompressed LWE decryption function, for testing purposes
mpz_class decryptLWE(std::vector<uint64_t> lwe_ct, std::vector<uint64_t> lwe_key, LWEParams &params) {
    uint64_t n = params.n;
    uint64_t p = params.p;
    mpz_class res = lwe_ct[n];
    for (uint64_t i = 0; i < n; i++) {
        res += (params.qBig - lwe_ct[i]) * lwe_key[i] % params.qBig;
    }
    res = res % params.qBig;
    res += params.qBig / (2 * p);
    res = res % params.qBig;
    res = res * p / params.qBig;
    return res;
}