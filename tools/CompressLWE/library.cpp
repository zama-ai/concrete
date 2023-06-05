#include "library.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <omp.h>

mpz_vec get_scales( const mpz_class& scale, size_t s){
    mpz_vec _scales(s);
    _scales[0] = 1;
    for (size_t i = 1; i < s; ++i) {
        _scales[i] = _scales[i - 1] * scale;
    }
    return _scales;
}

mpz_class compressSingle(
        mpz_vec &compressionKey,
        hcs::djcs::public_key &ahe_pk,
        std::vector<uint64_t> &lweCt,
        LWEParams &params) {

    uint64_t n = params.n;
    assert(lweCt.size() == n + 1); //LWE cipher is [a, b]

    // Construct a plaintext of -a
    // Multiply: (-a*E(sk))
    mpz_vec prod (n);
#pragma omp parallel for default(none) shared(ahe_pk, n, compressionKey, lweCt, prod, params) num_threads(omp_get_max_threads()/omp_get_num_threads()) schedule(static)
    for (uint64_t i = 0; i < n; i++) {
        mpz_class _a = params.qBig - lweCt[i];
        ahe_pk.ep_mul(prod[i], compressionKey[i], _a);
    }

    // Compute sum (over the batched ciphertext): sum(-a*E(sk))
    mpz_class sum = prod[0];
    for (size_t i = 1; i < n; ++i) {
        ahe_pk.ee_add(sum, sum, prod[i]);
    }

    // add b
    mpz_class b = lweCt[lweCt.size()-1];
    ahe_pk.ep_add(sum, sum, b);
    return sum;
}

mpz_class compressMPZCollapsed(
        mpz_vec &compressionKey,
        hcs::djcs::public_key &ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        const mpz_class& scale,
        LWEParams &params){
    uint64_t n = params.n;
    mpz_vec prod (n+1);

#pragma omp parallel for default(none) shared(n,cts,params, scale, ahe_pk, prod, compressionKey)
    for (size_t j = 0; j < n + 1; ++j) {
        // Scale-pack the ciphertexts, multiply by E(s)
        mpz_class lweCt_j;
        if (j == n) {
            lweCt_j = cts[0][n];
            for (size_t i = 1; i < cts.size(); ++i) {
                lweCt_j *= scale;
                lweCt_j += cts[i][n];
            }
            prod[n] = lweCt_j;
        }else {
            lweCt_j = params.qBig - cts[0][j];
            for (size_t i = 1; i < cts.size(); ++i) {
                lweCt_j *= scale;
                lweCt_j += (params.qBig - cts[i][j]);
            }
            ahe_pk.ep_mul(prod[j], compressionKey[j], lweCt_j);
        }
    }
    // Compute sum (over the batched ciphertext): sum(-a*E(sk))
    mpz_class sum = prod[0];
    for (size_t i = 1; i < n; ++i) {
        ahe_pk.ee_add(sum, sum, prod[i]);
    }
    ahe_pk.ep_add(sum, sum, prod[n]);
    return sum;
}

[[maybe_unused]] mpz_class compressMPZ(
        mpz_vec &compressionKey,
        hcs::djcs::public_key &ahe_pk,
        mpz_vec &lweCt,
        LWEParams &params) {

    uint64_t n = params.n;
    assert(lweCt.size() == n + 1); //LWE cipher is [a, b]

    // receive a scaled plaintext of -a (for a batch of ciphers)
    // Multiply: (-a*E(sk))
    mpz_vec prod (n);
#pragma omp parallel for default(none) shared(ahe_pk, n, compressionKey, lweCt, prod)
    for (uint64_t i = 0; i < n; i++) {
        ahe_pk.ep_mul(prod[i], compressionKey[i], lweCt[i]);
    }

    // Compute sum (over the batched ciphertext): sum(-a*E(sk))
    mpz_class sum = prod[0];
    for (size_t i = 1; i < n; ++i) {
        ahe_pk.ee_add(sum, sum, prod[i]);
    }

    // add b
    mpz_class b = lweCt[lweCt.size()-1];
    ahe_pk.ep_add(sum, sum, b);
    return sum;
}


[[maybe_unused]] mpz_vec scale_pack(
        std::vector<std::vector<uint64_t>> &cts,
        const mpz_class& scale,
        LWEParams &params){
    uint64_t n = params.n;
    mpz_vec ret(n + 1);
    mpz_vec _scales = get_scales(scale, cts.size());
#pragma omp parallel for default(none) shared(n,cts,ret, params, _scales)
    for (size_t j = 0; j < n + 1; ++j) {
        // Scale-pack the ciphertexts in ret
        ret[j] = 0;
        if (j == n) {
            for (size_t i = 0; i < cts.size(); ++i) {
                ret[n] += cts[i][n] * _scales[i];
            }
        }else {
            for (size_t i = 0; i < cts.size(); ++i) {
                ret[j] += (params.qBig - cts[i][j]) * _scales[i];
            }
        }
    }
    return ret;
}
CompressedCiphertext compressBatchedScaleFirst(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        LWEParams &params) {
    // Initialize an empty compressed ciphertext
    CompressedCiphertext c_ct(ahe_pk, params);

    // compute number of ahe ciphertexts needed
    uint64_t num_ahe_cts = std::ceil((double) cts.size() / (double) c_ct.maxCts);
    c_ct.ahe_cts.resize(num_ahe_cts);
    size_t _start =0, _end =c_ct.maxCts, ciphers = cts.size();
    // construct every ahe ciphertext
    for (uint64_t i = 0; i < num_ahe_cts; i++) {
        _end = std::min(_end, ciphers);
        std::vector<std::vector<uint64_t>> to_compress (cts.begin() + _start, cts.begin()+_end);
        // assign the new ahe ct
        c_ct.ahe_cts[i] = compressMPZCollapsed(compressionKey, ahe_pk, to_compress, c_ct.scale, params);
        _start = _end;
        _end+=c_ct.maxCts;
    }
    return c_ct;
}


CompressedCiphertext compressBatchedEncryptFirst(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        LWEParams &params) {
    omp_set_max_active_levels(2);
    omp_set_dynamic(0);
    // Initialize an empty compressed ciphertext
    CompressedCiphertext c_ct(ahe_pk, params);

    // compute number of ahe ciphertexts needed
    uint64_t num_ahe_cts = std::ceil((double) cts.size() / (double) c_ct.maxCts);
    c_ct.ahe_cts.resize(num_ahe_cts);
    // construct every ahe ciphertext
#pragma omp parallel for default(none) shared(ahe_pk, num_ahe_cts, compressionKey, c_ct, cts, params) num_threads(num_ahe_cts) schedule(static)
    for (uint64_t i = 0; i < num_ahe_cts; i++) {
        // start by compressing the first ciphertext
        mpz_class p_ct = compressSingle(compressionKey, ahe_pk, cts[i * c_ct.maxCts], params);
        for (uint64_t j = 1; j < c_ct.maxCts; j++) {
            // check if we compressed everything, we break
            if (i * c_ct.maxCts + j >= cts.size())
                break;
            // scale the ahe ciphertext to leave room for the next lwe ct in the lower bits
            ahe_pk.ep_mul(p_ct, p_ct, c_ct.scale);
            mpz_class curr = compressSingle(compressionKey, ahe_pk, cts[i * c_ct.maxCts + j], params);
            // add the compressed lwe ct to the lower bits
            ahe_pk.ee_add(p_ct, p_ct, curr);
        }
        // assign the new ahe ct
        c_ct.ahe_cts[i] = p_ct;
    }
    return c_ct;
}

CompressedCiphertext compressBatched(
        mpz_vec &compressionKey,
        hcs::djcs::public_key& ahe_pk,
        std::vector<std::vector<uint64_t>> &cts,
        LWEParams &params) {
    return compressBatchedEncryptFirst(compressionKey, ahe_pk,cts ,params);
}

Keys* generateKeys(std::vector<uint64_t> &lweKey, size_t s) {
    Keys *ret;
    ret = new Keys(lweKey.size());
    // Generate a key pair with modulus of size 2048 bits
    hcs::djcs::generate_key_pair(ret->ahe_pk, ret->ahe_sk, s, 2048);
#pragma omp parallel for default(none) shared(lweKey, ret)
    for(size_t i=0;i<lweKey.size();i++){
        ret->compKey[i] = lweKey[i];
        ret->ahe_pk.encrypt(ret->compKey[i], ret->compKey[i]);
    }
    return ret;

}

mpz_class decryptCompressedSingle(
        mpz_class &resultCt,
        hcs::djcs::private_key &ahe_sk,
        LWEParams &params) {
    mpz_class res;
    ahe_sk.decrypt(res, resultCt);
    finalizeDecryption(res, params);
    return res;

}

mpz_vec decryptCompressedBatched(
        CompressedCiphertext &compressedCiphertext,
        hcs::djcs::private_key &ahe_sk,
        LWEParams &params,
        uint64_t ciphers,
        bool reverse){

    mpz_class scale_mpz = compressedCiphertext.scale;
    mpz_vec all_results;

    uint64_t remaining = ciphers;
    // iterate over ahe ciphertexts
    for (mpz_class &result_ct: compressedCiphertext.ahe_cts) {
        // decryptLWE and cast as in single
        mpz_class all_ciphers;
        ahe_sk.decrypt(all_ciphers, result_ct);
        // get num of lwe ciphers needed from this ahe ct
        uint64_t num_lwe_cts = std::min(remaining, compressedCiphertext.maxCts);
        std::vector<mpz_class> results(num_lwe_cts);
        // pack results one by one
        for (uint64_t i = 0; i < num_lwe_cts; i++) {
            // finalize decryption as in single
            mpz_class cipher = all_ciphers % scale_mpz;
            finalizeDecryption(cipher, params);
            results[i] = cipher;
            // unscale the plaintext, removing the ciphertext we just took
            all_ciphers = all_ciphers / scale_mpz;
        }
        // reverse because we compressed FILO
        if (reverse){
            std::reverse(results.begin(), results.end());
        }
        // append to final results
        all_results.insert(all_results.end(), results.begin(), results.end());
        // update needed ciphertexts from remaining ahe ciphers
        remaining -= num_lwe_cts;
    }
    return all_results;
}

void finalizeDecryption(mpz_class &res, LWEParams &params) {
    // lwe decryption rounding step
    res %= params.qBig;
    res += (params.qBig / (params.p * 2));
    if (res > params.qBig) {
        res -= params.qBig; //faster modularization for case of addition
    }
    res = (res * params.p) / params.qBig;
}