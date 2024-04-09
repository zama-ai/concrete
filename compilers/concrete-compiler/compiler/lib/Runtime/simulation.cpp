// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/simulation.h"
#include "concrete-cpu-noise-model.h"
#include "concrete-cpu.h"
#include "concrete/curves.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Runtime/wrappers.h"
#include "concretelang/Support/V0Parameters.h"
#include <assert.h>
#include <cmath>
#include <random>

using concretelang::csprng::SoftCSPRNG;

thread_local auto csprng = SoftCSPRNG(0);

const uint64_t UINT63_MAX = UINT64_MAX >> 1;

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
                               uint32_t glwe_dim, char *loc) {
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
  out = out + gaussian_noise(0, variance);
  if (out > UINT63_MAX) {
    printf("WARNING at %s: overflow happened during LUT in simulation\n", loc);
  }
  return out;
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
    uint32_t bsk_base_log, uint32_t polynomial_size, uint32_t pksk_base_log,
    uint32_t pksk_level_count, uint32_t glwe_dim) {

  // Check number of blocks
  assert(out_size == in_size && out_size == crt_decomp_size);

  uint64_t log_poly_size =
      static_cast<uint64_t>(ceil(log2(static_cast<double>(polynomial_size))));

  // Compute the numbers of bits to extract for each block and the total one.
  uint64_t total_number_of_bits_per_block = 0;
  auto number_of_bits_per_block = new uint64_t[crt_decomp_size]();
  for (uint64_t i = 0; i < crt_decomp_size; i++) {
    uint64_t modulus = crt_decomp_aligned[i + crt_decomp_offset];
    uint64_t nb_bit_to_extract =
        static_cast<uint64_t>(ceil(log2(static_cast<double>(modulus))));
    number_of_bits_per_block[i] = nb_bit_to_extract;

    total_number_of_bits_per_block += nb_bit_to_extract;
  }

  // Create the buffer of ciphertexts for storing the total number of bits to
  // extract.
  // The extracted bit should be in the following order:
  //
  // [msb(m%crt[n-1])..lsb(m%crt[n-1])...msb(m%crt[0])..lsb(m%crt[0])] where n
  // is the size of the crt decomposition
  auto extract_bits_output_buffer =
      new uint64_t[total_number_of_bits_per_block]{0};

  // Extraction of each bit for each block
  for (int64_t i = crt_decomp_size - 1, extract_bits_output_offset = 0; i >= 0;
       extract_bits_output_offset += number_of_bits_per_block[i--]) {
    auto nb_bits_to_extract = number_of_bits_per_block[i];

    size_t delta_log = 64 - nb_bits_to_extract;

    auto in_block = in_aligned[in_offset + i];

    // trick ( ct - delta/2 + delta/2^4  )
    uint64_t sub = (uint64_t(1) << (uint64_t(64) - nb_bits_to_extract - 1)) -
                   (uint64_t(1) << (uint64_t(64) - nb_bits_to_extract - 5));
    in_block -= sub;

    simulation_extract_bit_lwe_ciphertext_u64(
        &extract_bits_output_buffer[extract_bits_output_offset], in_block,
        delta_log, nb_bits_to_extract, log_poly_size, glwe_dim, lwe_small_dim,
        ksk_base_log, ksk_level_count, bsk_base_log, bsk_level_count, 64, 128);
  }

  size_t ct_in_count = total_number_of_bits_per_block;
  size_t lut_size = 1 << ct_in_count;
  size_t ct_out_count = out_size;
  size_t lut_count = ct_out_count;

  assert(lut_ct_size0 == lut_count);
  assert(lut_ct_size1 == lut_size);

  // Vertical packing
  simulation_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64(
      extract_bits_output_buffer, out_aligned + out_offset, ct_in_count,
      ct_out_count, lut_size, lut_count, lut_ct_aligned + lut_ct_offset,
      glwe_dim, log_poly_size, lwe_small_dim, bsk_level_count, bsk_base_log,
      cbs_level_count, cbs_base_log, pksk_level_count, pksk_base_log, 64, 128);
}

uint64_t sim_neg_lwe_u64(uint64_t plaintext) { return ~plaintext + 1; }

uint64_t sim_add_lwe_u64(uint64_t lhs, uint64_t rhs, char *loc) {
  if (lhs > UINT63_MAX - rhs) {
    printf("WARNING at %s: overflow happened during addition in simulation\n",
           loc);
  }
  return lhs + rhs;
}

uint64_t sim_mul_lwe_u64(uint64_t lhs, uint64_t rhs, char *loc) {
  if (rhs != 0 && lhs > UINT63_MAX / rhs) {
    printf("WARNING at %s: overflow happened during multiplication in "
           "simulation\n",
           loc);
  }
  return lhs * rhs;
}

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
