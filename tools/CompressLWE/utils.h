//
// Created by r5akhava on 4/17/23.
//

#ifndef COMPRESSLWE_UTILS_H
#define COMPRESSLWE_UTILS_H

#include <cmath>
#include "ipcl/ipcl.hpp"
#include "defines.h"


BigNumber from64(const uint64_t &num);

uint64_t to64(const BigNumber &num);

std::vector<BigNumber> from64(std::vector<uint64_t> nums);

//helper to sample a random uint64_t
uint64_t sample(uint64_t log_q);

// uncompressed LWE decryption function, for testing purposes
uint64_t decryptLWE(std::vector<uint64_t> lwe_ct, std::vector<uint64_t> lwe_key, const LWEParams &params);

#endif //COMPRESSLWE_UTILS_H
