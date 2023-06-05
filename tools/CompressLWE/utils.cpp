//
// Created by r5akhava on 4/17/23.
//

#include "utils.h"

BigNumber from64(const uint64_t &num) {
    char hex[20];
    sprintf(hex, "%ld", num);
    return {hex};
}

uint64_t to64(const BigNumber &num) {
    std::string hex;
    num.num2hex(hex);
    uint64_t ret = strtoull(hex.c_str(), nullptr, 16);
    return ret;
}

std::vector<BigNumber> from64(std::vector<uint64_t> nums) {
    std::vector<BigNumber> ret(nums.size());
    for (uint64_t i = 0; i < nums.size(); ++i) {
        ret[i] = from64(nums[i]);
    }
    return ret;
}

//helper to sample a random uint64_t
uint64_t sample(uint64_t log_q) {
    uint64_t n = ((uint64_t) rand() << 32) ^ ((uint64_t) rand());
    if (log_q < 64)
        return n % (1UL << log_q);
    else
        return n;
}

// uncompressed LWE decryption function, for testing purposes
uint64_t decryptLWE(std::vector<uint64_t> lwe_ct, std::vector<uint64_t> lwe_key, const LWEParams &params) {
    uint64_t n = params.n;
    uint64_t p = params.p;
    BigNumber res = from64(lwe_ct[n]);
    for (int i = 0; i < n; i++) {
        res += (params.qBig - from64(lwe_ct[i])) * from64(lwe_key[i]) % params.qBig;
    }
    res = res % params.qBig;
    res += params.qBig / (2 * p);
    res = res % params.qBig;
    res = res * p / params.qBig;
    return to64(res);
}