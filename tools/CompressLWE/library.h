#ifndef LIBRARY_H
#define LIBRARY_H

#include "utils.h"

mpz_class compressSingle(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<uint64_t> &lweCt,
        LWEParams &params);

[[maybe_unused]] mpz_class compressMPZ(
        mpz_vec &compressionKey,
        hcs::djcs::public_key &ahe_pk,
        mpz_vec &lweCt,
        LWEParams &params);

CompressedCiphertext compressBatchedEncryptFirst(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        LWEParams &params);

CompressedCiphertext compressBatchedScaleFirst(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        LWEParams &params);

CompressedCiphertext compressBatched(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        LWEParams &params);

mpz_class decryptCompressedSingle(
        mpz_class &resultCt,
        hcs::djcs::private_key& ahe_sk,
        LWEParams &params);

mpz_vec decryptCompressedBatched(
        CompressedCiphertext &compressedCiphertext,
        hcs::djcs::private_key &ahe_sk,
        LWEParams &params,
        uint64_t ciphers,
        bool reverse=true);

Keys* generateKeys(std::vector<uint64_t> &lweKey, size_t s =3);

void finalizeDecryption(
        mpz_class &res,
        LWEParams &params);

#endif //LIBRARY_H