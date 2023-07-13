#include "compress_lwe/library.h"
#include "compress_lwe/defines.h"
#include "libhcs++/djcs.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <gmpxx.h>
#include <iostream>
#include <libhcs++.hpp>
#include <memory>
#include <omp.h>
#include <sstream>
#include <vector>

namespace comp {

mpz_class *to_raw(const mpz &a) { return (mpz_class *)a.ptr; }

uint64_t to_uint64_t(mpz_class &res) {
  assert(res < (1_mpz << 64));

  mpz_class high2 = res >> 32;

  uint32_t high = mpz_get_ui(high2.get_mpz_t());

  mpz_class low2 = res - (high2 << 32);

  uint32_t low = mpz_get_ui(low2.get_mpz_t());

  uint64_t result = ((uint64_t)high) << 32;

  result += low;

  return result;
}

mpz_vec get_scales(const mpz &scale, size_t s) {
  mpz_vec _scales(s);
  *((mpz_class *)_scales[0].ptr) = 1_mpz;
  for (size_t i = 1; i < s; ++i) {
    *((mpz_class *)_scales[i].ptr) =
        *((mpz_class *)_scales[i - 1].ptr) * *((mpz_class *)scale.ptr);
  }
  return _scales;
}

mpz compressSingle(comp::CompressionKey &compressionKey, const uint64_t *lweCt,
                   uint64_t lwe_dim) {

  hcs::djcs::public_key &ahe_pk = *compressionKey.ahe_pk->ptr;

  // Construct a plaintext of -a
  // Multiply: (-a*E(sk))
  std::vector<mpz_class> prod(lwe_dim);
#pragma omp parallel for default(none)                                         \
    shared(ahe_pk, lwe_dim, compressionKey, lweCt, prod)                       \
    num_threads(omp_get_max_threads() / omp_get_num_threads())                 \
    schedule(static)
  for (uint64_t i = 0; i < lwe_dim; i++) {
    uint64_t minus = -lweCt[i];
    mpz_class _a = minus;
    ahe_pk.ep_mul(prod[i], *to_raw(compressionKey.compKey[i]), _a);
  }

  // Compute sum (over the batched ciphertext): sum(-a*E(sk))
  mpz_class sum = lwe_dim > 0 ? prod[0] : 1;

  for (size_t i = 1; i < lwe_dim; ++i) {
    ahe_pk.ee_add(sum, sum, prod[i]);
  }

  // add b
  mpz_class b = lweCt[lwe_dim];
  ahe_pk.ep_add(sum, sum, b);
  return mpz(new mpz_class(sum));
}

mpz compressMPZCollapsed(comp::CompressionKey &compressionKey,
                         const uint64_t *cts, uint64_t lwe_dim,
                         uint64_t ct_count, const mpz &scale_) {

  mpz_class scale = *to_raw(scale_);

  std::vector<mpz_class> prod(lwe_dim + 1);

  mpz_class qBig = 1_mpz << 64;

  hcs::djcs::public_key &ahe_pk = *compressionKey.ahe_pk->ptr;

#pragma omp parallel for default(none)                                         \
    shared(lwe_dim, ct_count, cts, qBig, scale, ahe_pk, prod, compressionKey)
  for (size_t j = 0; j < lwe_dim + 1; ++j) {
    // Scale-pack the ciphertexts, multiply by E(s)
    mpz_class lweCt_j;
    if (j == lwe_dim) {
      lweCt_j = cts[lwe_dim];
      for (size_t i = 1; i < ct_count; ++i) {
        lweCt_j *= scale;
        lweCt_j += cts[(lwe_dim + 1) * i + lwe_dim];
      }
      prod[lwe_dim] = lweCt_j;
    } else {
      lweCt_j = qBig - cts[j];
      for (size_t i = 1; i < ct_count; ++i) {
        lweCt_j *= scale;
        lweCt_j += (qBig - cts[(lwe_dim + 1) * i + j]);
      }
      ahe_pk.ep_mul(prod[j], *to_raw(compressionKey.compKey[j]), lweCt_j);
    }
  }
  // Compute sum (over the batched ciphertext): sum(-a*E(sk))
  mpz_class sum = prod[0];
  for (size_t i = 1; i < lwe_dim; ++i) {
    ahe_pk.ee_add(sum, sum, prod[i]);
  }
  ahe_pk.ep_add(sum, sum, prod[lwe_dim]);
  return mpz(new mpz_class(sum));
}

[[maybe_unused]] mpz compressMPZ(comp::CompressionKey &compressionKey,
                                 mpz_vec &lweCt) {

  hcs::djcs::public_key &ahe_pk = *compressionKey.ahe_pk->ptr;

  uint64_t n = compressionKey.compKey.size();
  assert(lweCt.size() == n + 1); // LWE cipher is [a, b]

  // receive a scaled plaintext of -a (for a batch of ciphers)
  // Multiply: (-a*E(sk))
  std::vector<mpz_class> prod(n);
#pragma omp parallel for default(none)                                         \
    shared(ahe_pk, n, compressionKey, lweCt, prod)
  for (uint64_t i = 0; i < n; i++) {
    ahe_pk.ep_mul(prod[i], *to_raw(compressionKey.compKey[i]),
                  *to_raw(lweCt[i]));
  }

  // Compute sum (over the batched ciphertext): sum(-a*E(sk))
  mpz_class sum = prod[0];
  for (size_t i = 1; i < n; ++i) {
    ahe_pk.ee_add(sum, sum, prod[i]);
  }

  // add b
  mpz_class b = *to_raw(lweCt[lweCt.size() - 1]);
  ahe_pk.ep_add(sum, sum, b);
  return mpz(new mpz_class(sum));
}

[[maybe_unused]] mpz_vec scale_pack(const uint64_t *cts, uint64_t lwe_dim,
                                    uint64_t ct_count, const mpz &scale) {
  mpz_vec ret;
  mpz_vec _scales = get_scales(scale, ct_count);
  mpz_class qBig = 1;
  qBig <<= 64;

#pragma omp parallel for default(none)                                         \
    shared(lwe_dim, ct_count, cts, ret, _scales, qBig)
  for (size_t j = 0; j < lwe_dim + 1; ++j) {
    // Scale-pack the ciphertexts in ret
    mpz_class next = 0;
    if (j == lwe_dim) {
      for (size_t i = 0; i < ct_count; ++i) {
        next += cts[(lwe_dim + 1) * i + lwe_dim] * *to_raw(_scales[i]);
      }
    } else {
      for (size_t i = 0; i < ct_count; ++i) {
        next += (qBig - cts[(lwe_dim + 1) * i + j]) * *to_raw(_scales[i]);
      }
    }
    ret.push_back(mpz(new mpz_class(next)));
  }
  return ret;
}
CompressedCiphertext
compressBatchedScaleFirst(comp::CompressionKey &compressionKey,
                          const uint64_t *cts, uint64_t lwe_dim,
                          uint64_t ct_count) {

  // Initialize an empty compressed ciphertext
  CompressedCiphertext c_ct(compressionKey.ahe_pk, lwe_dim);

  // compute number of ahe ciphertexts needed
  uint64_t num_ahe_cts = std::ceil((double)ct_count / (double)c_ct.maxCts);
  c_ct.ahe_cts.resize(num_ahe_cts);
  size_t _start = 0, _end = c_ct.maxCts, ciphers = ct_count;
  // construct every ahe ciphertext
  for (uint64_t i = 0; i < num_ahe_cts; i++) {
    _end = std::min(_end, ciphers);
    const uint64_t *to_compress = cts + (lwe_dim + 1) * _start;
    // assign the new ahe ct
    *to_raw(c_ct.ahe_cts[i]) = *to_raw(compressMPZCollapsed(
        compressionKey, to_compress, lwe_dim, _end - _start, c_ct.scale));
    _start = _end;
    _end += c_ct.maxCts;
  }
  return c_ct;
}

CompressedCiphertext
compressBatchedEncryptFirst(comp::CompressionKey &compressionKey,
                            const uint64_t *cts, uint64_t lwe_dim,
                            uint64_t ct_count) {

  omp_set_max_active_levels(2);
  omp_set_dynamic(0);

  hcs::djcs::public_key &ahe_pk = *compressionKey.ahe_pk->ptr;

  // Initialize an empty compressed ciphertext
  CompressedCiphertext c_ct(compressionKey.ahe_pk,
                            compressionKey.compKey.size());

  // compute number of ahe ciphertexts needed
  uint64_t num_ahe_cts = std::ceil((double)ct_count / (double)c_ct.maxCts);
  c_ct.ahe_cts.resize(num_ahe_cts);
  // construct every ahe ciphertext
#pragma omp parallel for default(none)                                         \
    shared(ahe_pk, num_ahe_cts, compressionKey, c_ct, cts, lwe_dim, ct_count)  \
    num_threads(num_ahe_cts) schedule(static)
  for (uint64_t i = 0; i < num_ahe_cts; i++) {
    // start by compressing the
    // first ciphertext
    mpz p_ct = compressSingle(compressionKey,
                              cts + (lwe_dim + 1) * i * c_ct.maxCts, lwe_dim);
    for (uint64_t j = 1; j < c_ct.maxCts; j++) {
      // check if we compressed
      // everything, we break
      if (i * c_ct.maxCts + j >= ct_count)
        break;
      // scale the ahe ciphertext
      // to leave room for the
      // next lwe ct in the lower
      // bits
      ahe_pk.ep_mul(*to_raw(p_ct), *to_raw(p_ct), *to_raw(c_ct.scale));
      mpz curr = compressSingle(
          compressionKey, cts + (lwe_dim + 1) * (i * c_ct.maxCts + j), lwe_dim);
      // add the compressed lwe
      // ct to the lower bits
      ahe_pk.ee_add(*to_raw(p_ct), *to_raw(p_ct), *to_raw(curr));
    }
    // assign the new ahe ct
    c_ct.ahe_cts[i].ptr = new mpz_class(*to_raw(p_ct));
  }
  return c_ct;
}

CompressedCiphertext compressBatched(comp::CompressionKey &compressionKey,
                                     const uint64_t *cts, uint64_t lwe_dim,
                                     uint64_t ct_count) {
  return compressBatchedEncryptFirst(compressionKey, cts, lwe_dim, ct_count);
}

FullKeys generateKeys(const std::vector<uint64_t> &lweKey, size_t s) {
  FullKeys ret;

  // ret.compKey = mpz_vec(lweKey.size());
  for (int i = 0; i < lweKey.size(); i++) {
    ret.compKey.push_back(mpz(new mpz_class(0)));
  }

  auto hr_ = std::make_shared<hcs::random>();

  ret.ahe_pk = std::make_shared<PublicKey>(new hcs::djcs::public_key(hr_));
  ret.ahe_sk = std::make_shared<PrivateKey>(new hcs::djcs::private_key(hr_));

  hcs::djcs::generate_key_pair(*(ret.ahe_pk->ptr), *(ret.ahe_sk->ptr), s, 2048);

#pragma omp parallel for default(none) shared(lweKey, ret)
  for (size_t i = 0; i < lweKey.size(); i++) {
    *to_raw(ret.compKey[i]) = lweKey[i];
    ret.ahe_pk->ptr->encrypt(*to_raw(ret.compKey[i]), *to_raw(ret.compKey[i]));
  }
  return ret;
}

uint64_t decryptCompressedSingle(mpz &resultCt, PrivateKey &ahe_sk_) {

  hcs::djcs::private_key &ahe_sk = *ahe_sk_.ptr;

  mpz res(new mpz_class);

  ahe_sk.decrypt(*to_raw(res), *to_raw(resultCt));

  mpz_class modulus = *to_raw(res) % (1_mpz << 64);

  return to_uint64_t(modulus);
}

std::vector<uint64_t>
decryptCompressedBatched(CompressedCiphertext &compressedCiphertext,
                         PrivateKey &ahe_sk, uint64_t ciphers
                         //  ,bool reverse
) {

  mpz_class scale_mpz = *to_raw(compressedCiphertext.scale);
  std::vector<uint64_t> all_results;

  uint64_t remaining = ciphers;
  // iterate over ahe ciphertexts
  for (mpz &result_ct : compressedCiphertext.ahe_cts) {
    // decryptLWE and cast as in
    // single
    mpz_class all_ciphers;
    ahe_sk.ptr->decrypt(all_ciphers, *to_raw(result_ct));
    // get num of lwe ciphers
    // needed from this ahe ct
    uint64_t num_lwe_cts = std::min(remaining, compressedCiphertext.maxCts);
    std::vector<mpz> results(num_lwe_cts);

    all_results.resize(all_results.size() + num_lwe_cts);

    // pack results one by one
    for (uint64_t i = 0; i < num_lwe_cts; i++) {
      mpz_class scale = 1;
      scale <<= 64;

      mpz_class a = all_ciphers % scale;

      all_results[all_results.size() - i - 1] = to_uint64_t(a);
      // unscale the plaintext,
      // removing the ciphertext
      // we just took
      all_ciphers = all_ciphers / scale_mpz;
    }
    // update needed ciphertexts
    // from remaining ahe ciphers
    remaining -= num_lwe_cts;
  }

  return all_results;
}

} // namespace comp
