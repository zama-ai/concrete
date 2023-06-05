#ifndef COMPRESSLWE_DEFINES_H
#define COMPRESSLWE_DEFINES_H

#include <cmath>
#include <utility>
#include <vector>
#include <libhcs++.hpp>

typedef std::vector<mpz_class> mpz_vec;

struct Keys {
    hcs::random hr;
    hcs::djcs::public_key ahe_pk = hcs::djcs::public_key(hr);
    hcs::djcs::private_key ahe_sk = hcs::djcs::private_key(hr);
    mpz_vec compKey;
    explicit Keys(size_t n=630) {
        compKey = mpz_vec(n);
    }

};

struct LWEParams {
    uint64_t n; // Ciphertext dimension of underlying lwe scheme
    uint64_t logQ;
    uint64_t p; // Plaintext modulus of underlying lwe scheme
    mpz_class qBig;
    bool binaryKeys;

    LWEParams(uint64_t n, uint64_t log_q, uint64_t p, bool binaryKeys=false) : n(n), logQ(log_q), p(p), binaryKeys(binaryKeys) {
        qBig = 1_mpz << log_q;
    }

};

struct CompressedCiphertext {
    mpz_class scale{};
    mpz_vec ahe_cts;
    hcs::djcs::public_key ahe_pk;
    LWEParams lweParams;
    uint64_t maxCts;

    CompressedCiphertext(hcs::djcs::public_key &pk, LWEParams & lweParams): ahe_pk(pk), lweParams(lweParams) {
        uint64_t bitwidth, s;
        s = ahe_pk.as_ptr()->s;
        if (lweParams.binaryKeys){
            bitwidth = lweParams.logQ;
        }else{
            bitwidth = 2 * lweParams.logQ;
        }
        double ahe_capacity = 2047.0*s;
        uint64_t logScale = (uint64_t) ceil(log2(lweParams.n + 1) + bitwidth);
        scale = 1_mpz << logScale;
        maxCts = (uint64_t) std::floor(ahe_capacity / (double) logScale);
    }
};

#endif //COMPRESSLWE_DEFINES_H