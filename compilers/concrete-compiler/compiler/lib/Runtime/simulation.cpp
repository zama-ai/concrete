// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/simulation.h"
#include "concrete-cpu-noise-model.h"
#include "concrete-cpu.h"
#include "concrete/curves.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/Runtime/wrappers.h"
#include "concretelang/Support/V0Parameters.h"
#include <assert.h>
#include <cmath>
#include <random>

inline concrete::SecurityCurve *security_curve() {
  return concrete::getSecurityCurve(128, concrete::BINARY);
}

uint64_t from_torus(double torus) {
  assert(torus >= 0 && torus < 1 && "torus value must be in [0, 1)");
  return (uint64_t)round(torus * pow(2, 64));
}

// TODO: what's the overhead of creating a csprng everytime? Should we have a
// single one?
uint64_t gaussian_noise(double mean, double variance) {
  uint64_t random_gaussian_buff[2];
  auto csprng = concretelang::clientlib::ConcreteCSPRNG(0);
  concrete_cpu_fill_with_random_gaussian(random_gaussian_buff, 2, variance,
                                         csprng.ptr);
  return random_gaussian_buff[0];
}

uint64_t sim_encrypt_lwe_u64(uint64_t message, uint32_t lwe_dim, void *csprng) {
  double variance = security_curve()->getVariance(1, lwe_dim, 64);
  uint64_t random_gaussian_buff[2];
  concrete_cpu_fill_with_random_gaussian(random_gaussian_buff, 2, variance,
                                         (Csprng *)csprng);
  uint64_t encryption_noise = random_gaussian_buff[0];
  return message + encryption_noise;
}

uint64_t sim_keyswitch_lwe_u64(uint64_t plaintext, uint32_t level,
                               uint32_t base_log, uint32_t input_lwe_dim,
                               uint32_t output_lwe_dim) {
  double variance_ksk = security_curve()->getVariance(1, output_lwe_dim, 64);
  double variance = concrete_cpu_variance_keyswitch(input_lwe_dim, base_log,
                                                    level, 64, variance_ksk);
  uint64_t ks_noise = gaussian_noise(0, variance);
  return plaintext + ks_noise;
}

uint64_t sim_bootstrap_lwe_u64(uint64_t plaintext, uint64_t *tlu_allocated,
                               uint64_t *tlu_aligned, uint64_t tlu_offset,
                               uint64_t tlu_size, uint64_t tlu_stride,
                               uint32_t input_lwe_dim, uint32_t poly_size,
                               uint32_t level, uint32_t base_log,
                               uint32_t glwe_dim) {
  auto tlu = tlu_aligned + tlu_offset;

  // modulus switching
  double variance_ms =
      concrete_cpu_estimate_modulus_switching_noise_with_binary_key(
          input_lwe_dim, log2(poly_size), 64);
  uint64_t shift = (64 - log2(poly_size) - 2);
  // mod_switch noise
  auto noise = gaussian_noise(0, variance_ms);
  noise >>= shift;
  noise += noise & 1;
  noise >>= 1;
  // mod_switch
  uint64_t mod_switched = plaintext >> shift;
  mod_switched += mod_switched & 1;
  mod_switched >>= 1;
  mod_switched += noise;
  mod_switched %= 2 * poly_size;

  uint64_t out;
  // blind rotate & sample extract:
  // instead of doing a plynomial multiplication, then extracting the first
  // coeff, we directly extract the appropriate coeff from the tlu.
  if (mod_switched < poly_size)
    out = tlu[mod_switched];
  else
    out = -tlu[mod_switched % poly_size];

  double variance_bsk = security_curve()->getVariance(glwe_dim, poly_size, 64);
  double variance = concrete_cpu_variance_blind_rotate(
      input_lwe_dim, glwe_dim, poly_size, base_log, level, 64,
      mlir::concretelang::optimizer::DEFAULT_FFT_PRECISION, variance_bsk);
  return out + gaussian_noise(0, variance);
}

void sim_wop_pbs_crt(
    // Output 1D memref
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride,
    // Input 1D memref
    uint64_t *in_allocated, uint64_t *in_aligned, uint64_t in_offset,
    uint64_t in_size, uint64_t in_stride,
    // clear text lut 2D memref
    uint64_t *lut_ct_allocated, uint64_t *lut_ct_aligned,
    uint64_t lut_ct_offset, uint64_t lut_ct_size0, uint64_t lut_ct_size1,
    uint64_t lut_ct_stride0, uint64_t lut_ct_stride1,
    // CRT decomposition 1D memref
    uint64_t *crt_decomp_allocated, uint64_t *crt_decomp_aligned,
    uint64_t crt_decomp_offset, uint64_t crt_decomp_size,
    uint64_t crt_decomp_stride,
    // Additional crypto parameters
    uint32_t lwe_small_dim, uint32_t cbs_level_count, uint32_t cbs_base_log,
    uint32_t ksk_level_count, uint32_t ksk_base_log, uint32_t bsk_level_count,
    uint32_t bsk_base_log, uint32_t fpksk_level_count, uint32_t fpksk_base_log,
    uint32_t polynomial_size) {
  // TODO
}

uint64_t sim_neg_lwe_u64(uint64_t plaintext) { return ~plaintext + 1; }

void sim_encode_expand_lut_for_boostrap(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *in_allocated,
    uint64_t *in_aligned, uint64_t in_offset, uint64_t in_size,
    uint64_t in_stride, uint32_t poly_size, uint32_t output_bits,
    bool is_signed) {
  return memref_encode_expand_lut_for_bootstrap(
      out_allocated, out_aligned, out_offset, out_size, out_stride,
      in_allocated, in_aligned, in_offset, in_size, in_stride, poly_size,
      output_bits, is_signed);
}

void sim_encode_plaintext_with_crt(uint64_t *output_allocated,
                                   uint64_t *output_aligned,
                                   uint64_t output_offset, uint64_t output_size,
                                   uint64_t output_stride, uint64_t input,
                                   uint64_t *mods_allocated,
                                   uint64_t *mods_aligned, uint64_t mods_offset,
                                   uint64_t mods_size, uint64_t mods_stride,
                                   uint64_t mods_product) {
  return memref_encode_plaintext_with_crt(
      output_allocated, output_aligned, output_offset, output_size,
      output_stride, input, mods_allocated, mods_aligned, mods_offset,
      mods_size, mods_stride, mods_product);
}

void sim_encode_lut_for_crt_woppbs(
    // Output encoded/expanded lut
    uint64_t *output_lut_allocated, uint64_t *output_lut_aligned,
    uint64_t output_lut_offset, uint64_t output_lut_size0,
    uint64_t output_lut_size1, uint64_t output_lut_stride0,
    uint64_t output_lut_stride1,
    // Input lut
    uint64_t *input_lut_allocated, uint64_t *input_lut_aligned,
    uint64_t input_lut_offset, uint64_t input_lut_size,
    uint64_t input_lut_stride,
    // Crt coprimes
    uint64_t *crt_decomposition_allocated, uint64_t *crt_decomposition_aligned,
    uint64_t crt_decomposition_offset, uint64_t crt_decomposition_size,
    uint64_t crt_decomposition_stride,
    // Crt number of bits
    uint64_t *crt_bits_allocated, uint64_t *crt_bits_aligned,
    uint64_t crt_bits_offset, uint64_t crt_bits_size, uint64_t crt_bits_stride,
    // Crypto parameters
    uint32_t modulus_product, bool is_signed) {
  return memref_encode_lut_for_crt_woppbs(
      output_lut_allocated, output_lut_aligned, output_lut_offset,
      output_lut_size0, output_lut_size1, output_lut_stride0,
      output_lut_stride1, input_lut_allocated, input_lut_aligned,
      input_lut_offset, input_lut_size, input_lut_stride,
      crt_decomposition_allocated, crt_decomposition_aligned,
      crt_decomposition_offset, crt_decomposition_size,
      crt_decomposition_stride, crt_bits_allocated, crt_bits_aligned,
      crt_bits_offset, crt_bits_size, crt_bits_stride, modulus_product,
      is_signed);
}
