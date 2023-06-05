#include "library.h"
#include <cassert>
#include <cmath>

ipcl::CipherText compressSingle(
        const ipcl::CipherText &compressionKey,
        const std::vector<uint64_t> &lweCt,
        const LWEParams &params) {

    uint64_t n = params.n;
    assert(lweCt.size() == n + 1); //LWE cipher is [a, b]

    // Paillier public parameters
    const ipcl::PublicKey &a_pk = *(compressionKey.getPubKey());
    const BigNumber &sq = *(a_pk.getNSQ());

    // Construct a plaintext of -a
    std::vector<BigNumber> _a(n);
    for (uint64_t i = 0; i < n; i++) {
        _a[i] = params.qBig - from64(lweCt[i]);
    }
    ipcl::PlainText _a_pt(_a);

    // Multiply: (-a*E(sk))
    ipcl::CipherText prod = _a_pt * compressionKey;

    // Compute sum (over the batched ciphertext): sum(-a*E(sk))
    auto sum = prod[0];
    for (size_t i = 1; i < n; ++i) {
        sum = sum * prod[i] % sq; // paillier for addition: [multiply coeffs]
    }
    // cast as a ciphertext
    auto ret = ipcl::CipherText(a_pk, sum);

    // add b
    auto b_pt = ipcl::PlainText(from64(lweCt[lweCt.size() - 1]));
    ret = ret + b_pt;
    return ret;

}

CompressedCiphertext compressBatched(
        const ipcl::CipherText &compressionKey,
        const std::vector<std::vector<uint64_t>> &cts,
        const LWEParams &params) {

    // Initialize an empty compressed ciphertext
    CompressedCiphertext c_ct(params.n, params.logQ, params.p);

    // cast ciphertext width as a paillier plaintext
    BigNumber scale_bn = c_ct.scale;
    ipcl::PlainText scale_pt(scale_bn);

    // compute number of paillier ciphertexts needed
    uint64_t num_paillier_cts = std::ceil((float) cts.size() / (float) c_ct.maxCts);
    c_ct.pCts.resize(num_paillier_cts);

    // construct every paillier ciphertext
    for (uint64_t i = 0; i < num_paillier_cts; i++) {
        // start by compressing the first ciphertext
        auto p_ct = compressSingle(compressionKey, cts[i * c_ct.maxCts], params);
        for (uint64_t j = 1; j < c_ct.maxCts; j++) {
            // check if we compressed everything, we break
            if (i * c_ct.maxCts + j >= cts.size())
                break;
            // scale the paillier ciphertext to leave room for the next lwe ct in the lower bits
            p_ct = p_ct * scale_pt;
            // add the compressed lwe ct to the lower bits
            p_ct = p_ct + compressSingle(compressionKey, cts[i * c_ct.maxCts + j], params);
        }
        // assign the new paillier ct
        c_ct.pCts[i] = p_ct;
    }
    return c_ct;
}

Keys generateKeys(const std::vector<uint64_t> &lweKey) {
    ipcl::KeyPair paiKeys = ipcl::generateKeypair(2048, true);
    //TODO: switch modulus of lweKey here
    ipcl::PlainText sk_pt(from64(lweKey));
    ipcl::CipherText compressionKey = paiKeys.pub_key.encrypt(sk_pt);
    return {paiKeys, compressionKey};

}

uint64_t decryptCompressedSingle(
        const ipcl::CipherText &resultCt,
        const ipcl::PrivateKey &paiSk,
        const LWEParams &params) {
    BigNumber res = paiSk.decrypt(resultCt);
    finalizeDecryption(res, params);
    return to64(res);

}

std::vector<uint64_t> decryptCompressedBatched(
        const CompressedCiphertext &compressedCiphertext,
        const ipcl::PrivateKey &paiSk,
        const LWEParams &params, uint64_t ciphers) {

    BigNumber scale_bn = compressedCiphertext.scale;
    std::vector<uint64_t> all_results;

    uint64_t remaining = ciphers;
    // iterate over paillier ciphertexts
    for (const ipcl::CipherText &result_ct: compressedCiphertext.pCts) {
        // decryptLWE and cast as in single
        ipcl::PlainText r_pt = paiSk.decrypt(result_ct);
        BigNumber all_ciphers = r_pt;
        // get num of lwe ciphers needed from this paillier ct
        uint64_t num_lwe_cts = std::min(remaining, compressedCiphertext.maxCts);
        std::vector<uint64_t> results(num_lwe_cts);
        // pack results one by one
        for (uint64_t i = 0; i < num_lwe_cts; i++) {
            // finalize decryption as in single
            BigNumber cipher = all_ciphers % scale_bn;
            finalizeDecryption(cipher, params);
            results[i] = to64(cipher);
            // unscale the plaintext, removing the ciphertext we just took
            all_ciphers = all_ciphers / scale_bn;
        }
        // reverse because we compressed FILO
        std::reverse(results.begin(), results.end());
        // append to final results
        all_results.insert(all_results.end(), results.begin(), results.end());
        // update needed ciphertexts from remaining paillier ciphers
        remaining -= num_lwe_cts;
    }
    return all_results;
}

void finalizeDecryption(BigNumber &res, const LWEParams &params) {
    // lwe decryption rounding step
    res %= params.qBig;
    res += (params.qBig / from64(params.p * 2));
    if (res > params.qBig) {
        res -= params.qBig; //faster modularization for case of addition
    }
    res = (res * from64(params.p)) / params.qBig;
}