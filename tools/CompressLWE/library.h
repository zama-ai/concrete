#ifndef LIBRARY_H
#define LIBRARY_H

#include "defines.h"
#include "utils.h"

PaiCiphertext compressSingle(const PaiCiphertext &compressionKey,
                             const uint64_t *lweCt, const LWEParams &params);

CompressedCiphertext compressBatched(const PaiCiphertext &compressionKey,
                                     const uint64_t *cts, uint64_t cts_count,
                                     const LWEParams &params);

uint64_t decryptCompressedSingle(const PaiCiphertext &resultCt,
                                 const PaiPrivateKey &paiSk,
                                 const LWEParams &params);

std::vector<uint64_t>
decryptCompressedBatched(const CompressedCiphertext &compressedCiphertext,
                         const PaiPrivateKey &paiSk, const LWEParams &params,
                         uint64_t ciphers);

PaiFullKeys generateKeys(const std::vector<uint64_t> &lweKey);

#endif // LIBRARY_H
