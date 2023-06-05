#ifndef COMPRESSLWE_DEFINES_H
#define COMPRESSLWE_DEFINES_H

#include <cmath>
#include <utility>
#include "ipcl/ipcl.hpp"

struct Keys {
    ipcl::KeyPair paiKeys; //Paillier keypair
    ipcl::CipherText compKey; //A vector of paillier ciphers
};

struct LWEParams {
    uint64_t n; // Ciphertext dimension of underlying lwe scheme
    uint64_t p; // Plaintext modulus of underlying lwe scheme
    uint64_t logQ;
    BigNumber qBig;

    LWEParams(uint64_t n, uint64_t log_q, uint64_t p) : n(n), logQ(log_q), p(p) {
        this->qBig = 1;
        for (uint64_t i = 0; i < log_q; i++) {
            this->qBig *= 2;
        }
    }
};

struct CompressedCiphertext {
    BigNumber scale; // width of every lwe cipher in packed paillier cipher
    uint64_t paiBitLen = 2048;

    uint64_t logScale;
    uint64_t maxCts;

    std::vector<ipcl::CipherText> pCts;

    CompressedCiphertext(uint64_t n, uint64_t log_q, uint64_t p) {
        logScale = (uint64_t) ceil(log2(n + 1) + 2 * log_q);
        scale = 1;
        for (uint64_t i = 0; i < logScale; i++) {
            scale *= 2;
        }
        maxCts = (uint64_t) std::floor((float) paiBitLen / logScale);
    }
};

#endif //COMPRESSLWE_DEFINES_H
