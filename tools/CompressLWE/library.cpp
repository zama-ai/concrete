#include "library.h"
#include "defines.h"

#include "ipcl/ipcl.hpp"

PaiCiphertext compressSingle(const PaiCiphertext &compressionKey,
                             const uint64_t *lweCt, const LWEParams &params) {

  uint64_t n = params.n;

  // Paillier public parameters
  const ipcl::PublicKey &a_pk = *compressionKey.ptr->getPubKey();
  const BigNumber &sq = *(a_pk.getNSQ());

  // Construct a plaintext of -a
  std::vector<BigNumber> _a(n);
  for (uint64_t i = 0; i < n; i++) {
    _a[i] = *params.qBig.ptr - from64(lweCt[i]);
  }
  ipcl::PlainText _a_pt(_a);

  // Multiply: (-a*E(sk))
  ipcl::CipherText prod = _a_pt * *compressionKey.ptr;

  // Compute sum (over the batched ciphertext): sum(-a*E(sk))
  auto sum = prod[0];
  for (size_t i = 1; i < n; ++i) {
    sum = sum * prod[i] % sq; // paillier for addition: [multiply coeffs]
  }
  // cast as a ciphertext
  auto ret = new ipcl::CipherText(a_pk, sum);

  // add b
  auto b_pt = ipcl::PlainText(from64(lweCt[n]));
  *ret = *ret + b_pt;
  return PaiCiphertext(ret);
}

CompressedCiphertext compressBatched(const PaiCiphertext &compressionKey,
                                     const uint64_t *cts, uint64_t cts_count,
                                     const LWEParams &params) {

  // Initialize an empty compressed ciphertext
  CompressedCiphertext c_ct(params.n, params.logQ);

  // cast ciphertext width as a paillier plaintext
  const BigNumber &scale_bn = *c_ct.scale.ptr;

  ipcl::PlainText scale_pt(scale_bn);

  // compute number of paillier ciphertexts needed
  uint64_t num_paillier_cts = std::ceil((float)cts_count / (float)c_ct.maxCts);

  // construct every paillier ciphertext
  for (uint64_t i = 0; i < num_paillier_cts; i++) {
    // start by compressing the first ciphertext
    PaiCiphertext p_ct = compressSingle(
        compressionKey, cts + (params.n + 1) * i * c_ct.maxCts, params);

    for (uint64_t j = 1; j < c_ct.maxCts; j++) {
      // check if we compressed everything, we break
      if (i * c_ct.maxCts + j >= cts_count)
        break;
      // scale the paillier ciphertext to leave room for the next lwe ct in the
      // lower bits
      *p_ct.ptr = *p_ct.ptr * scale_pt;
      // add the compressed lwe ct to the lower bits
      *p_ct.ptr =
          *p_ct.ptr +
          *compressSingle(compressionKey,
                          cts + (params.n + 1) * (i * c_ct.maxCts + j), params)
               .ptr;
    }
    // assign the new paillier ct
    c_ct.pCts.push_back(std::move(p_ct));
  }
  return c_ct;
}

PaiFullKeys generateKeys(const std::vector<uint64_t> &lweKey) {
  ipcl::KeyPair paiKeys = ipcl::generateKeypair(2048, true);
  // TODO: switch modulus of lweKey here
  ipcl::PlainText sk_pt(from64(lweKey));
  ipcl::CipherText compressionKey = paiKeys.pub_key.encrypt(sk_pt);

  return PaiFullKeys{
      .pub_key = std::make_shared<PaiPublicKey>(
          PaiPublicKey(new ipcl::PublicKey(paiKeys.pub_key))),
      .priv_key = std::make_shared<PaiPrivateKey>(
          PaiPrivateKey(new ipcl::PrivateKey(paiKeys.priv_key))),
      .compKey = std::make_shared<PaiCiphertext>(
          PaiCiphertext(new ipcl::CipherText(compressionKey))),
  };
}

uint64_t decryptCompressedSingle(const PaiCiphertext &resultCt,
                                 const PaiPrivateKey &paiSk,
                                 const LWEParams &params) {

  BigNumber_ res(new BigNumber(paiSk.ptr->decrypt(*resultCt.ptr)));

  *res.ptr %= *params.qBig.ptr;

  return to64(*res.ptr);
}

std::vector<uint64_t>
decryptCompressedBatched(const CompressedCiphertext &compressedCiphertext,
                         const PaiPrivateKey &paiSk, const LWEParams &params,
                         uint64_t ciphers) {

  const BigNumber &scale_bn = *compressedCiphertext.scale.ptr;
  std::vector<uint64_t> all_results;

  uint64_t remaining = ciphers;
  // iterate over paillier ciphertexts
  for (const PaiCiphertext &result_ct : compressedCiphertext.pCts) {

    // decryptLWE and cast as in single
    ipcl::PlainText r_pt = paiSk.ptr->decrypt(*result_ct.ptr);
    BigNumber all_ciphers = r_pt;
    // get num of lwe ciphers needed from this paillier ct
    uint64_t num_lwe_cts = std::min(remaining, compressedCiphertext.maxCts);
    std::vector<uint64_t> results(num_lwe_cts);
    // pack results one by one
    for (uint64_t i = 0; i < num_lwe_cts; i++) {
      // finalize decryption as in single

      BigNumber *cipher(new BigNumber());
      *cipher = all_ciphers % scale_bn;

      results[i] = to64(*cipher % *params.qBig.ptr);
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
