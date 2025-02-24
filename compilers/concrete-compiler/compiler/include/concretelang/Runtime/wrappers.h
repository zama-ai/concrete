// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_WRAPPERS_H
#define CONCRETELANG_RUNTIME_WRAPPERS_H

#include "concrete-cpu.h"
#include <complex.h>
#include <concretelang/Runtime/context.h>
#include <stddef.h>
#include <stdint.h>

extern "C" {

/// \brief Expands the input LUT
///
/// It duplicates values as needed to fill mega cases, taking care of the
/// encoding and the half mega case shift in the process as well. All sizes
/// should be powers of 2.
///
/// \param output where to write the expanded LUT
/// \param output_size
/// \param out_MESSAGE_BITS number of bits of message to be used
/// \param lut original LUT
/// \param lut_size
void memref_encode_expand_lut_for_bootstrap(
    uint64_t *output_lut_allocated, uint64_t *output_lut_aligned,
    uint64_t output_lut_offset, uint64_t output_lut_size,
    uint64_t output_lut_stride, uint64_t *input_lut_allocated,
    uint64_t *input_lut_aligned, uint64_t input_lut_offset,
    uint64_t input_lut_size, uint64_t input_lut_stride, uint32_t poly_size,
    uint32_t out_MESSAGE_BITS, bool is_signed);

void memref_encode_lut_for_crt_woppbs(
    uint64_t *output_lut_allocated, uint64_t *output_lut_aligned,
    uint64_t output_lut_offset, uint64_t output_lut_size0,
    uint64_t output_lut_size1, uint64_t output_lut_stride0,
    uint64_t output_lut_stride1, uint64_t *input_lut_allocated,
    uint64_t *input_lut_aligned, uint64_t input_lut_offset,
    uint64_t input_lut_size, uint64_t input_lut_stride,
    uint64_t *crt_decomposition_allocated, uint64_t *crt_decomposition_aligned,
    uint64_t crt_decomposition_offset, uint64_t crt_decomposition_size,
    uint64_t crt_decomposition_stride, uint64_t *crt_bits_allocated,
    uint64_t *crt_bits_aligned, uint64_t crt_bits_offset,
    uint64_t crt_bits_size, uint64_t crt_bits_stride, uint32_t modulus_product,
    bool is_signed);

void memref_encode_plaintext_with_crt(
    uint64_t *output_allocated, uint64_t *output_aligned,
    uint64_t output_offset, uint64_t output_size, uint64_t output_stride,
    uint64_t input, uint64_t *mods_allocated, uint64_t *output_lut_aligned,
    uint64_t mods_offset, uint64_t output_lut_size, uint64_t mods_stride,
    uint64_t mods_product);

void memref_add_lwe_ciphertexts_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *ct1_allocated, uint64_t *ct1_aligned,
    uint64_t ct1_offset, uint64_t ct1_size, uint64_t ct1_stride);

void memref_add_plaintext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t plaintext);

void memref_mul_cleartext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t cleartext);

void memref_negate_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride);

void memref_keyswitch_lwe_u64(uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride,
                              uint32_t level, uint32_t base_log,
                              uint32_t input_lwe_dim, uint32_t output_lwe_dim,
                              uint32_t ksk_index,
                              mlir::concretelang::RuntimeContext *context);

void memref_batched_add_lwe_ciphertexts_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *ct1_allocated,
    uint64_t *ct1_aligned, uint64_t ct1_offset, uint64_t ct1_size0,
    uint64_t ct1_size1, uint64_t ct1_stride0, uint64_t ct1_stride1);

void memref_batched_add_plaintext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *ct1_allocated,
    uint64_t *ct1_aligned, uint64_t ct1_offset, uint64_t ct1_size,
    uint64_t ct1_stride);

void memref_batched_add_plaintext_cst_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t plaintext);

void memref_batched_mul_cleartext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *ct1_allocated,
    uint64_t *ct1_aligned, uint64_t ct1_offset, uint64_t ct1_size,
    uint64_t ct1_stride);

void memref_batched_mul_cleartext_cst_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t cleartext);

void memref_batched_negate_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1);

void memref_batched_keyswitch_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint32_t level,
    uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    uint32_t ksk_index, mlir::concretelang::RuntimeContext *context);

void *memref_keyswitch_async_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, mlir::concretelang::RuntimeContext *context);

void memref_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);

void memref_batched_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size,
    uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t poly_size,
    uint32_t level, uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);

void memref_batched_mapped_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size0,
    uint64_t tlu_size1, uint64_t tlu_stride0, uint64_t tlu_stride1,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);

void *memref_bootstrap_async_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);

void memref_await_future(uint64_t *out_allocated, uint64_t *out_aligned,
                         uint64_t out_offset, uint64_t out_size,
                         uint64_t out_stride, void *future,
                         uint64_t *in_allocated, uint64_t *in_aligned,
                         uint64_t in_offset, uint64_t in_size,
                         uint64_t in_stride);

uint64_t encode_crt(int64_t plaintext, uint64_t modulus, uint64_t product);

void memref_wop_pbs_crt_buffer(
    // Output memref 2D memref
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size_0, uint64_t out_size_1, uint64_t out_stride_0,
    uint64_t out_stride_1,
    // Input memref
    uint64_t *in_allocated, uint64_t *in_aligned, uint64_t in_offset,
    uint64_t in_size_0, uint64_t in_size_1, uint64_t in_stride_0,
    uint64_t in_stride_1,
    // clear text lut
    uint64_t *lut_ct_allocated, uint64_t *lut_ct_aligned,
    uint64_t lut_ct_offset, uint64_t lut_ct_size0, uint64_t lut_ct_size1,
    uint64_t lut_ct_stride0, uint64_t lut_ct_stride1,
    // CRT decomposition
    uint64_t *crt_decomp_allocated, uint64_t *crt_decomp_aligned,
    uint64_t crt_decomp_offset, uint64_t crt_decomp_size,
    uint64_t crt_decomp_stride,
    // Additional crypto parameters
    uint32_t lwe_small_size, uint32_t cbs_level_count, uint32_t cbs_base_log,
    uint32_t ksk_level_count, uint32_t ksk_base_log, uint32_t bsk_level_count,
    uint32_t bsk_base_log, uint32_t fpksk_level_count, uint32_t fpksk_base_log,
    uint32_t polynomial_size,
    // Key indices
    uint32_t ksk_index, uint32_t bsk_index, uint32_t pksk_index,
    // runtime context that hold evaluation keys
    mlir::concretelang::RuntimeContext *context);

void memref_copy_one_rank(uint64_t *src_allocated, uint64_t *src_aligned,
                          uint64_t src_offset, uint64_t src_size,
                          uint64_t src_stride, uint64_t *dst_allocated,
                          uint64_t *dst_aligned, uint64_t dst_offset,
                          uint64_t dst_size, uint64_t dst_stride);

// Single ciphertext CUDA functions ///////////////////////////////////////////

/// \brief Run Keyswitch on GPU.
///
/// It handles memory copy of the different arguments from CPU to GPU, and
/// freeing memory.
void memref_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t ksk_index,
    mlir::concretelang::RuntimeContext *context);

/// \brief Run bootstrapping on GPU.
///
/// It handles memory copy of the different arguments from CPU to GPU, and
/// freeing memory.
void memref_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);

// Batched CUDA function //////////////////////////////////////////////////////

void memref_batched_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint32_t level,
    uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    uint32_t ksk_index, mlir::concretelang::RuntimeContext *context);

void memref_batched_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size,
    uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t poly_size,
    uint32_t level, uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);

void memref_batched_mapped_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size0,
    uint64_t tlu_size1, uint64_t tlu_stride0, uint64_t tlu_stride1,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context);
// Tracing ////////////////////////////////////////////////////////////////////
void memref_trace_ciphertext(uint64_t *ct0_allocated, uint64_t *ct0_aligned,
                             uint64_t ct0_offset, uint64_t ct0_size,
                             uint64_t ct0_stride, char *message_ptr,
                             uint32_t message_len, uint32_t msb);

void memref_trace_plaintext(uint64_t input, uint64_t input_width,
                            char *message_ptr, uint32_t message_len,
                            uint32_t msb);

void memref_trace_message(char *message_ptr, uint32_t message_len);

/// @brief Allocate memory using malloc and check for nullptr
/// @param size number of bytes to allocate
/// @return pointer to the allocated memory or nullptr
void *concrete_checked_malloc(size_t size);
}

#endif
