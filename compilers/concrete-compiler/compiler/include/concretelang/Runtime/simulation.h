// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_SIMULATION_H
#define CONCRETELANG_RUNTIME_SIMULATION_H

#include <stdint.h>

extern "C" {

/// \brief simulate the encryption of a value by adding noise
///
/// \param message encoded message to encrypt
/// \param lwe_dim
/// \return noisy plaintext
uint64_t sim_encrypt_lwe_u64(uint64_t message, uint32_t lwe_dim);

/// \brief simulate the negation of a noisy plaintext
///
/// \param plaintext noisy plaintext
/// \return uint64_t
uint64_t sim_neg_lwe_u64(uint64_t plaintext);

/// \brief simulate a keyswitch on a noisy plaintext
///
/// \param plaintext noisy plaintext
/// \param level
/// \param base_log
/// \param input_lwe_dim
/// \param output_lwe_dim
/// \return uint64_t
uint64_t sim_keyswitch_lwe_u64(uint64_t plaintext, uint32_t level,
                               uint32_t base_log, uint32_t input_lwe_dim,
                               uint32_t output_lwe_dim);

/// \brief simulate a bootstrap on a noisy plaintext
///
/// \param plaintext noisy plaintext
/// \param tlu_allocated
/// \param tlu_aligned
/// \param tlu_offset
/// \param tlu_size
/// \param tlu_stride
/// \param input_lwe_dim
/// \param poly_size
/// \param level
/// \param base_log
/// \param glwe_dim
/// \return uint64_t
uint64_t sim_bootstrap_lwe_u64(uint64_t plaintext, uint64_t *tlu_allocated,
                               uint64_t *tlu_aligned, uint64_t tlu_offset,
                               uint64_t tlu_size, uint64_t tlu_stride,
                               uint32_t input_lwe_dim, uint32_t poly_size,
                               uint32_t level, uint32_t base_log,
                               uint32_t glwe_dim);

void sim_encode_expand_lut_for_boostrap(
    uint64_t *in_allocated, uint64_t *in_aligned, uint64_t in_offset,
    uint64_t in_size, uint64_t in_stride, uint64_t *out_allocated,
    uint64_t *out_aligned, uint64_t out_offset, uint64_t out_size,
    uint64_t out_stride, uint32_t poly_size, uint32_t output_bits,
    bool is_signed);
}

void sim_encode_plaintext_with_crt(uint64_t *output_allocated,
                                   uint64_t *output_aligned,
                                   uint64_t output_offset, uint64_t output_size,
                                   uint64_t output_stride, uint64_t input,
                                   uint64_t *mods_allocated,
                                   uint64_t *mods_aligned, uint64_t mods_offset,
                                   uint64_t mods_size, uint64_t mods_stride,
                                   uint64_t mods_product);

#endif
