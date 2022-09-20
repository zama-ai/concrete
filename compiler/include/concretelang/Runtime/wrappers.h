// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_WRAPPERS_H
#define CONCRETELANG_RUNTIME_WRAPPERS_H

#include "concretelang/Runtime/context.h"

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
void encode_and_expand_lut(uint64_t *output, size_t output_size,
                           size_t out_MESSAGE_BITS, const uint64_t *lut,
                           size_t lut_size);

void memref_expand_lut_in_trivial_glwe_ct_u64(
    uint64_t *glwe_ct_allocated, uint64_t *glwe_ct_aligned,
    uint64_t glwe_ct_offset, uint64_t glwe_ct_size, uint64_t glwe_ct_stride,
    uint32_t poly_size, uint32_t glwe_dimension, uint32_t out_precision,
    uint64_t *lut_allocated, uint64_t *lut_aligned, uint64_t lut_offset,
    uint64_t lut_size, uint64_t lut_stride);

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
                              mlir::concretelang::RuntimeContext *context);
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
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context);

void *memref_bootstrap_async_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
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
    uint64_t lut_ct_offset, uint64_t lut_ct_size, uint64_t lut_ct_stride,
    // CRT decomposition
    uint64_t *crt_decomp_allocated, uint64_t *crt_decomp_aligned,
    uint64_t crt_decomp_offset, uint64_t crt_decomp_size,
    uint64_t crt_decomp_stride,
    // Additional crypto parameters
    uint32_t lwe_small_size, uint32_t cbs_level_count, uint32_t cbs_base_log,
    uint32_t polynomial_size,
    // runtime context that hold evluation keys
    mlir::concretelang::RuntimeContext *context);

void memref_copy_one_rank(uint64_t *src_allocated, uint64_t *src_aligned,
                          uint64_t src_offset, uint64_t src_size,
                          uint64_t src_stride, uint64_t *dst_allocated,
                          uint64_t *dst_aligned, uint64_t dst_offset,
                          uint64_t dst_size, uint64_t dst_stride);

/// \brief Run bootstrapping on GPU.
///
/// It handles memory copy of the different arguments from CPU to GPU, and
/// freeing memory.
///
/// \param out_allocated
/// \param out_aligned
/// \param out_offset
/// \param out_size
/// \param out_stride
/// \param ct0_allocated
/// \param ct0_aligned
/// \param ct0_offset
/// \param ct0_size
/// \param ct0_stride
/// \param tlu_allocated
/// \param tlu_aligned
/// \param tlu_offset
/// \param tlu_size
/// \param tlu_stride
/// \param input_lwe_dim LWE input dimension
/// \param poly_size polynomial size
/// \param level level
/// \param base_log base log
/// \param glwe_dim
/// \param precision
/// \param context
void memref_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context);

/// \brief Run Keyswitch on GPU.
///
/// It handles memory copy of the different arguments from CPU to GPU, and
/// freeing memory.
///
/// \param out_allocated
/// \param out_aligned
/// \param out_offset
/// \param out_size
/// \param out_stride
/// \param ct0_allocated
/// \param ct0_aligned
/// \param ct0_offset
/// \param ct0_size
/// \param ct0_stride
/// \param level
/// \param base_log
/// \param input_lwe_dim LWE input dimension
/// \param output_lwe_dim LWE output dimension
/// \param context
void memref_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    mlir::concretelang::RuntimeContext *context);

/// \brief Copy ciphertext from CPU to GPU.
///
/// It handles memory allocation on GPU.
///
/// \param ct_allocated
/// \param ct_aligned
/// \param ct_offset
/// \param ct_size
/// \param ct_stride
/// \param gpu_idx index of the GPU to use
/// \param stream cuda stream to use for the copy
/// \return void* pointer to the GPU ciphertext
void *memcpy_async_ct_to_gpu(uint64_t *ct_allocated, uint64_t *ct_aligned,
                             uint64_t ct_offset, uint64_t ct_size,
                             uint64_t ct_stride, uint32_t gpu_idx,
                             void *stream);

/// \brief Copy ciphertext from GPU to CPU.
///
/// Memory on GPU won't be freed after the copy.
///
/// \param out_allocated
/// \param out_aligned
/// \param out_offset
/// \param out_size
/// \param out_stride
/// \param ct_gpu
/// \param size
/// \param gpu_idx index of the GPU to use
/// \param stream cuda stream to use for the copy
void memcpy_async_ct_to_cpu(uint64_t *out_allocated, uint64_t *out_aligned,
                            uint64_t out_offset, uint64_t out_size,
                            uint64_t out_stride, void *ct_gpu, size_t size,
                            uint32_t gpu_idx, void *stream);

/// \brief Copy bootstrapping key from CPU to GPU.
///
/// It handles memory allocation on GPU, as well as conversion to the Fourier
/// domain.
///
/// \param context
/// \param input_lwe_dim
/// \param poly_size
/// \param level
/// \param glwe_dim
/// \param gpu_idx index of the GPU to use
/// \param stream cuda stream to use for the copy
/// \return void*  pointer to the GPU bsk
void *memcpy_async_bsk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t input_lwe_dim, uint32_t poly_size,
                              uint32_t level, uint32_t glwe_dim,
                              uint32_t gpu_idx, void *stream);

/// \brief Copy keyswitching key from CPU to GPU.
///
/// It handles memory allocation on GPU.
///
/// \param context
/// \param level
/// \param input_lwe_dim
/// \param output_lwe_dim
/// \param gpu_idx index of the GPU to use
/// \param stream cuda stream to use for the copy
/// \return void*  pointer to the GPU ksk
void *memcpy_async_ksk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t level, uint32_t input_lwe_dim,
                              uint32_t output_lwe_dim, uint32_t gpu_idx,
                              void *stream);

/// \brief Free gpu memory.
///
/// \param gpu_ptr pointer to the GPU memory to free
/// \param gpu_idx index of the GPU to use
void free_from_gpu(void *gpu_ptr, uint32_t gpu_idx);
}
#endif
