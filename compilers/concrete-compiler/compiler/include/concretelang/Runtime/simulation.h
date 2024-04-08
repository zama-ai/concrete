// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_SIMULATION_H
#define CONCRETELANG_RUNTIME_SIMULATION_H

#include <stdint.h>

extern "C" {

/// \brief simulate the encryption of a value by adding noise
///
/// \param message encoded message to encrypt
/// \param lwe_dim
/// \param csprng used to generate noise during encryption
/// \return noisy plaintext
uint64_t sim_encrypt_lwe_u64(uint64_t message, uint32_t lwe_dim, void *csprng);

/// \brief simulate the negation of a noisy plaintext
///
/// \param plaintext noisy plaintext
/// \return uint64_t
uint64_t sim_neg_lwe_u64(uint64_t plaintext);

/// \brief simulate the addition of a noisy plaintext with another
/// plaintext (noisy or not)
///
/// The function also checks for overflow and print a warning when it happens
///
/// \param lhs left operand
/// \param rhs right operand
/// \param loc
/// \return uint64_t
uint64_t sim_add_lwe_u64(uint64_t lhs, uint64_t rhs, char *loc);

/// \brief simulate the multiplication of a noisy plaintext with an integer
///
/// The function also checks for overflow and print a warning when it happens
///
/// \param lhs left operand
/// \param rhs right operand
/// \param loc
/// \return uint64_t
uint64_t sim_mul_lwe_u64(uint64_t lhs, uint64_t rhs, char *loc);

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
/// \param loc
/// \return uint64_t
uint64_t sim_bootstrap_lwe_u64(uint64_t plaintext, uint64_t *tlu_allocated,
                               uint64_t *tlu_aligned, uint64_t tlu_offset,
                               uint64_t tlu_size, uint64_t tlu_stride,
                               uint32_t input_lwe_dim, uint32_t poly_size,
                               uint32_t level, uint32_t base_log,
                               uint32_t glwe_dim, char *loc);

/// simulate a WoP PBS
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
    uint32_t pksk_level_count, uint32_t glwe_dim);

void sim_encode_expand_lut_for_boostrap(
    uint64_t *in_allocated, uint64_t *in_aligned, uint64_t in_offset,
    uint64_t in_size, uint64_t in_stride, uint64_t *out_allocated,
    uint64_t *out_aligned, uint64_t out_offset, uint64_t out_size,
    uint64_t out_stride, uint32_t poly_size, uint32_t output_bits,
    bool is_signed);

void sim_encode_plaintext_with_crt(uint64_t *output_allocated,
                                   uint64_t *output_aligned,
                                   uint64_t output_offset, uint64_t output_size,
                                   uint64_t output_stride, uint64_t input,
                                   uint64_t *mods_allocated,
                                   uint64_t *mods_aligned, uint64_t mods_offset,
                                   uint64_t mods_size, uint64_t mods_stride,
                                   uint64_t mods_product);

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
    uint32_t modulus_product, bool is_signed);
}

#endif
