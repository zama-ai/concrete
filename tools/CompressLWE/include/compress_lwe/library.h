#ifndef COMPRESSLWE_LIBRARY_H
#define COMPRESSLWE_LIBRARY_H

#include "defines.h"
#include "utils.h"
#include <cstdint>
#include <vector>

namespace comp {
comp::mpz compressSingle(comp::CompressionKey &compressionKey,
                         const uint64_t *lweCt, uint64_t lwe_dim);

[[maybe_unused]] comp::mpz compressMPZ(comp::CompressionKey &compressionKey,
                                       comp::mpz_vec &lweCt);

comp::CompressedCiphertext
compressBatchedEncryptFirst(comp::CompressionKey &compressionKey,
                            const uint64_t *cts, uint64_t lwe_dim,
                            uint64_t ct_count);

comp::CompressedCiphertext
compressBatchedScaleFirst(comp::CompressionKey &compressionKey,
                          const uint64_t *cts, uint64_t lwe_dim,
                          uint64_t ct_count);

comp::CompressedCiphertext compressBatched(comp::CompressionKey &compressionKey,
                                           const uint64_t *cts,
                                           uint64_t lwe_dim, uint64_t ct_count);

uint64_t decryptCompressedSingle(comp::mpz &resultCt, comp::PrivateKey &ahe_sk);

std::vector<uint64_t>
decryptCompressedBatched(comp::CompressedCiphertext &compressedCiphertext,
                         comp::PrivateKey &ahe_sk, uint64_t ciphers);

comp::FullKeys generateKeys(const std::vector<uint64_t> &lweKey, size_t s = 3);

} // namespace comp

#endif // COMPRESSLWE_LIBRARY_H
