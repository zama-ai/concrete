#ifndef LIBRARY_H
#define LIBRARY_H

#include "utils.h"

ipcl::CipherText compressSingle(
        const ipcl::CipherText &compressionKey,
        const std::vector<uint64_t> &lweCt,
        const LWEParams &params);

CompressedCiphertext compressBatched(
        const ipcl::CipherText &compressionKey,
        const std::vector<std::vector<uint64_t>> &cts,
        const LWEParams &params);

uint64_t decryptCompressedSingle(
        const ipcl::CipherText &resultCt,
        const ipcl::PrivateKey &paiSk,
        const LWEParams &params);

std::vector<uint64_t> decryptCompressedBatched(
        const CompressedCiphertext &compressedCiphertext,
        const ipcl::PrivateKey &paiSk,
        const LWEParams &params,
        uint64_t ciphers);

Keys generateKeys(const std::vector<uint64_t> &lweKey);

void finalizeDecryption(
        BigNumber &res,
        const LWEParams &params);

#endif //LIBRARY_H