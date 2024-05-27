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
                               uint32_t glwe_dim, bool overflow_detection,
                               char *loc) {
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

  if (overflow_detection) {
    // get encoded info from lsb
    bool is_signed = (out >> 1) & 1;
    bool is_overflow = out & 1;
    // discard info bits (2 lsb)
    out = out & 18446744073709551612U;

    if (!is_signed && out > UINT63_MAX) {
      printf("WARNING at %s: overflow (padding bit) happened during LUT in "
             "simulation\n",
             loc);
    }
    if (is_overflow) {
      printf("WARNING at %s: overflow (original value didn't fit, so a modulus "
             "was applied) happened "
             "during LUT in "
             "simulation\n",
             loc);
    }
  }

  double variance_bsk = security_curve()->getVariance(glwe_dim, poly_size, 64);
  double variance = concrete_cpu_variance_blind_rotate(
      input_lwe_dim, glwe_dim, poly_size, base_log, level, 64,
      mlir::concretelang::optimizer::DEFAULT_FFT_PRECISION, variance_bsk);
  out = out + gaussian_noise(0, variance);
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

uint64_t sim_add_lwe_u64(uint64_t lhs, uint64_t rhs, char *loc,
                         bool is_signed) {
  const char msg_f[] =
      "WARNING at %s: overflow happened during addition in simulation\n";

  uint64_t result = lhs + rhs;

  if (is_signed) {
    // We shift left to discard the padding bit and only consider the message
    // for easier overflow checking
    int64_t lhs_signed = (int64_t)lhs << 1;
    int64_t rhs_signed = (int64_t)rhs << 1;
    if (lhs_signed > 0 && rhs_signed > INT64_MAX - lhs_signed)
      printf(msg_f, loc);
    else if (lhs_signed < 0 && rhs_signed < INT64_MIN - lhs_signed)
      printf(msg_f, loc);
  } else if (lhs > UINT63_MAX - rhs || result > UINT63_MAX) {
    printf(msg_f, loc);
  }
  return result;
}

uint64_t sim_mul_lwe_u64(uint64_t lhs, uint64_t rhs, char *loc,
                         bool is_signed) {
  const char msg_f[] =
      "WARNING at %s: overflow happened during multiplication in simulation\n";

  uint64_t result = lhs * rhs;

  if (is_signed) {
    // We shift left to discard the padding bit and only consider the message
    // for easier overflow checking
    int64_t lhs_signed = (int64_t)lhs << 1;
    int64_t rhs_signed = (int64_t)rhs << 1;
    if (lhs_signed != 0 && rhs_signed > INT64_MAX / lhs_signed)
      printf(msg_f, loc);
    else if (lhs_signed != 0 && rhs_signed < INT64_MIN / lhs_signed)
      printf(msg_f, loc);
  } else if (rhs != 0 && lhs > UINT63_MAX / rhs) {
    printf(msg_f, loc);
  }
  return result;
}

// a copy of memref_encode_expand_lut_for_bootstrap but which encodes overflow
// and sign info into the LUT. Those information should later be discarder by
// the LUT function
void sim_encode_expand_lut_for_boostrap(
    uint64_t *output_lut_allocated, uint64_t *output_lut_aligned,
    uint64_t output_lut_offset, uint64_t output_lut_size,
    uint64_t output_lut_stride, uint64_t *input_lut_allocated,
    uint64_t *input_lut_aligned, uint64_t input_lut_offset,
    uint64_t input_lut_size, uint64_t input_lut_stride, uint32_t poly_size,
    uint32_t out_MESSAGE_BITS, bool is_signed, bool overflow_detection) {

  assert(input_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                  "memref_encode_expand_lut_bootstrap");

  assert(output_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                   "memref_encode_expand_lut_bootstrap");

  size_t mega_case_size = output_lut_size / input_lut_size;

  assert((mega_case_size % 2) == 0);

  // flag for every element of the LUT to signal overflow
  std::vector<bool> overflow_info;
  // used to set the sign bit or not (2 is signed / 0 is not)
  uint64_t sign_bit_setter = 0;
  // compute overflow bit (if overflow detection is enabled)
  if (overflow_detection) {
    overflow_info = std::vector(output_lut_size, false);
    uint64_t upper_bound = uint64_t(1)
                           << (out_MESSAGE_BITS + (is_signed ? 1 : 0));
    for (size_t i = 0; i < input_lut_size; i++) {
      if (input_lut_aligned[input_lut_offset + i] >= upper_bound) {
        overflow_info[i] = true;
      } else {
        overflow_info[i] = false;
      }
    }
    // set the sign bit
    if (is_signed) {
      sign_bit_setter = 2;
    }
  }

  // When the bootstrap is executed on encrypted signed integers, the lut must
  // be half-rotated. This map takes care about properly indexing into the input
  // lut depending on what bootstrap gets executed.
  std::function<size_t(size_t)> indexMap;
  if (is_signed) {
    size_t halfInputSize = input_lut_size / 2;
    indexMap = [=](size_t idx) {
      if (idx < halfInputSize) {
        return idx + halfInputSize;
      } else {
        return idx - halfInputSize;
      }
    };
  } else {
    indexMap = [=](size_t idx) { return idx; };
  }

  // The first lut value should be centered over zero. This means that half of
  // it should appear at the beginning of the output lut, and half of it at the
  // end (but negated).
  for (size_t idx = 0; idx < mega_case_size / 2; ++idx) {
    output_lut_aligned[output_lut_offset + idx] =
        input_lut_aligned[input_lut_offset + indexMap(0)]
        << (64 - out_MESSAGE_BITS - 1);

    if (overflow_detection) {
      // set the sign bit
      output_lut_aligned[output_lut_offset + idx] |= sign_bit_setter;
      // set the overflow bit
      output_lut_aligned[output_lut_offset + idx] |= (uint64_t)overflow_info[0];
    }
  }
  for (size_t idx = (input_lut_size - 1) * mega_case_size + mega_case_size / 2;
       idx < output_lut_size; ++idx) {
    output_lut_aligned[output_lut_offset + idx] =
        -(input_lut_aligned[input_lut_offset + indexMap(0)]
          << (64 - out_MESSAGE_BITS - 1));

    if (overflow_detection) {
      // set the sign bit
      output_lut_aligned[output_lut_offset + idx] |= sign_bit_setter;
      // set the overflow bit
      output_lut_aligned[output_lut_offset + idx] |=
          (uint64_t)overflow_info[indexMap(0)];
    }
  }

  // Treats the other ut values.
  for (size_t lut_idx = 1; lut_idx < input_lut_size; ++lut_idx) {
    uint64_t lut_value = input_lut_aligned[input_lut_offset + indexMap(lut_idx)]
                         << (64 - out_MESSAGE_BITS - 1);
    if (overflow_detection) {
      // set the sign bit
      lut_value |= sign_bit_setter;
      // set the overflow bit
      lut_value |= (uint64_t)overflow_info[indexMap(lut_idx)];
    }

    size_t start = mega_case_size * (lut_idx - 1) + mega_case_size / 2;
    for (size_t output_idx = start; output_idx < start + mega_case_size;
         ++output_idx) {
      output_lut_aligned[output_lut_offset + output_idx] = lut_value;
    }
  }

  return;
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
