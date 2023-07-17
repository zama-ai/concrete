//
// Created by a2diaa on 4/17/23.
//

#ifndef COMPRESSLWE_UTILS_H
#define COMPRESSLWE_UTILS_H

#include <cmath>
#include "defines.h"


//helper to sample a random uint64_t
uint64_t sample(uint64_t log_q);

// uncompressed LWE decryption function, for testing purposes
mpz_class decryptLWE(std::vector<uint64_t> lwe_ct, std::vector<uint64_t> lwe_key, LWEParams &params);

#endif //COMPRESSLWE_UTILS_H
