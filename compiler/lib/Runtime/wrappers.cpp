// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/wrappers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/seeder.h"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static DefaultEngine *levelled_engine = nullptr;

DefaultEngine *get_levelled_engine() {
  if (levelled_engine == nullptr) {
    CAPI_ASSERT_ERROR(new_default_engine(best_seeder, &levelled_engine));
  }
  return levelled_engine;
}

#include "concretelang/ClientLib/CRT.h"
#include "concretelang/Runtime/wrappers.h"

#ifdef CONCRETELANG_CUDA_SUPPORT

// CUDA memory utils function /////////////////////////////////////////////////

void *memcpy_async_bsk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t input_lwe_dim, uint32_t poly_size,
                              uint32_t level, uint32_t glwe_dim,
                              uint32_t gpu_idx, void *stream) {
  return context->get_bsk_gpu(input_lwe_dim, poly_size, level, glwe_dim,
                              gpu_idx, stream);
}

void *memcpy_async_ksk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t level, uint32_t input_lwe_dim,
                              uint32_t output_lwe_dim, uint32_t gpu_idx,
                              void *stream) {
  return context->get_ksk_gpu(level, input_lwe_dim, output_lwe_dim, gpu_idx,
                              stream);
}

void *alloc_and_memcpy_async_to_gpu(uint64_t *buf_ptr, uint64_t buf_offset,
                                    uint64_t buf_size, uint32_t gpu_idx,
                                    void *stream) {
  size_t buf_size_ = buf_size * sizeof(uint64_t);
  void *ct_gpu = cuda_malloc(buf_size_, gpu_idx);
  cuda_memcpy_async_to_gpu(ct_gpu, buf_ptr + buf_offset, buf_size_, stream,
                           gpu_idx);
  return ct_gpu;
}

void memcpy_async_to_cpu(uint64_t *buf_ptr, uint64_t buf_offset,
                         uint64_t buf_size, void *buf_gpu, uint32_t gpu_idx,
                         void *stream) {
  cuda_memcpy_async_to_cpu(buf_ptr + buf_offset, buf_gpu,
                           buf_size * sizeof(uint64_t), stream, gpu_idx);
}

void free_from_gpu(void *gpu_ptr, uint32_t gpu_idx = 0) {
  cuda_drop(gpu_ptr, gpu_idx);
}

// Single ciphertext CUDA functions ///////////////////////////////////////////

void memref_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    mlir::concretelang::RuntimeContext *context) {
  assert(out_stride == 1);
  assert(ct0_stride == 1);
  memref_batched_keyswitch_lwe_cuda_u64(
      // Output 1D memref as 2D memref
      out_allocated, out_aligned, out_offset, 1, out_size, out_size, out_stride,
      // Output 1D memref as 2D memref
      ct0_allocated, ct0_aligned, ct0_offset, 1, ct0_size, ct0_size, ct0_stride,
      // Keyswitch additional arguments
      level, base_log, input_lwe_dim, output_lwe_dim, context);
}

void memref_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context) {
  memref_batched_bootstrap_lwe_cuda_u64(
      // Output 1D memref as 2D memref
      out_allocated, out_aligned, out_offset, 1, out_size, out_size, out_stride,
      // Input 1D memref as 2D memref
      ct0_allocated, ct0_aligned, ct0_offset, 1, ct0_size, ct0_size, ct0_stride,
      // Table lookup memref
      tlu_allocated, tlu_aligned, tlu_offset, tlu_size, tlu_stride,
      // Bootstrap additional arguments
      input_lwe_dim, poly_size, level, base_log, glwe_dim, precision, context);
}

// Batched CUDA function //////////////////////////////////////////////////////

void memref_batched_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint32_t level,
    uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    mlir::concretelang::RuntimeContext *context) {
  assert(out_size0 == ct0_size0);
  assert(out_size1 == output_lwe_dim + 1);
  assert(ct0_size1 == input_lwe_dim + 1);
  // TODO: Multi GPU
  uint32_t gpu_idx = 0;
  uint32_t num_samples = out_size0;
  uint64_t ct0_batch_size = ct0_size0 * ct0_size1;
  uint64_t out_batch_size = out_size0 * out_size1;

  // Create the cuda stream
  // TODO: Should be created by the compiler codegen
  void *stream = cuda_create_stream(gpu_idx);
  // Get the pointer on the keyswitching key on the GPU
  void *ksk_gpu = memcpy_async_ksk_to_gpu(context, level, input_lwe_dim,
                                          output_lwe_dim, gpu_idx, stream);
  // Move the input and output batch of ciphertexts to the GPU
  // TODO: The allocation should be done by the compiler codegen
  void *ct0_gpu = alloc_and_memcpy_async_to_gpu(
      ct0_aligned, ct0_offset, ct0_batch_size, gpu_idx, stream);
  void *out_gpu = alloc_and_memcpy_async_to_gpu(
      out_aligned, out_offset, out_batch_size, gpu_idx, stream);
  // Run the keyswitch kernel on the GPU
  cuda_keyswitch_lwe_ciphertext_vector_64(
      stream, gpu_idx, out_gpu, ct0_gpu, ksk_gpu, input_lwe_dim, output_lwe_dim,
      base_log, level, num_samples);
  // Copy the output batch of ciphertext back to CPU
  memcpy_async_to_cpu(out_aligned, out_offset, out_batch_size, out_gpu, gpu_idx,
                      stream);
  cuda_synchronize_device(gpu_idx);
  // free memory that we allocated on gpu
  cuda_drop(ct0_gpu, gpu_idx);
  cuda_drop(out_gpu, gpu_idx);
  cuda_destroy_stream(stream, gpu_idx);
}

void memref_batched_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size,
    uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t poly_size,
    uint32_t level, uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context) {
  assert(out_size0 == ct0_size0);
  // TODO: Multi GPU
  uint32_t gpu_idx = 0;
  uint32_t num_samples = out_size0;
  uint64_t ct0_batch_size = ct0_size0 * ct0_size1;
  uint64_t out_batch_size = out_size0 * out_size1;

  // Create the cuda stream
  // TODO: Should be created by the compiler codegen
  void *stream = cuda_create_stream(gpu_idx);
  // Get the pointer on the bootstraping key on the GPU
  void *fbsk_gpu = memcpy_async_bsk_to_gpu(context, input_lwe_dim, poly_size,
                                           level, glwe_dim, gpu_idx, stream);
  // Move the input and output batch of ciphertext to the GPU
  // TODO: The allocation should be done by the compiler codegen
  void *ct0_gpu = alloc_and_memcpy_async_to_gpu(
      ct0_aligned, ct0_offset, ct0_batch_size, gpu_idx, stream);
  void *out_gpu = alloc_and_memcpy_async_to_gpu(
      out_aligned, out_offset, out_batch_size, gpu_idx, stream);

  // Construct the glwe accumulator (on CPU)
  // TODO: Should be done outside of the bootstrap call, compile time if
  // possible. Refactor in progress
  uint64_t glwe_ct_len = poly_size * (glwe_dim + 1);
  uint64_t glwe_ct_size = glwe_ct_len * sizeof(uint64_t);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size);

  CAPI_ASSERT_ERROR(
      default_engine_discard_trivially_encrypt_glwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), glwe_ct, glwe_ct_len, tlu_aligned + tlu_offset,
          poly_size));

  // Move the glwe accumulator to the GPU
  void *glwe_ct_gpu =
      alloc_and_memcpy_async_to_gpu(glwe_ct, 0, glwe_ct_len, gpu_idx, stream);

  // Move test vector indexes to the GPU, the test vector indexes is set of 0
  uint32_t num_test_vectors = 1, lwe_idx = 0,
           test_vector_idxes_size = num_samples * sizeof(uint32_t);
  void *test_vector_idxes = malloc(test_vector_idxes_size);
  memset(test_vector_idxes, 0, test_vector_idxes_size);
  void *test_vector_idxes_gpu = cuda_malloc(test_vector_idxes_size, gpu_idx);
  cuda_memcpy_async_to_gpu(test_vector_idxes_gpu, test_vector_idxes,
                           test_vector_idxes_size, stream, gpu_idx);
  // Run the bootstrap kernel on the GPU
  cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
      stream, gpu_idx, out_gpu, glwe_ct_gpu, test_vector_idxes_gpu, ct0_gpu,
      fbsk_gpu, input_lwe_dim, glwe_dim, poly_size, base_log, level,
      num_samples, num_test_vectors, lwe_idx,
      cuda_get_max_shared_memory(gpu_idx));
  // Copy the output batch of ciphertext back to CPU
  memcpy_async_to_cpu(out_aligned, out_offset, out_batch_size, out_gpu, gpu_idx,
                      stream);
  cuda_synchronize_device(gpu_idx);
  // Free the glwe accumulator (on CPU)
  free(glwe_ct);
  // free memory that we allocated on gpu
  cuda_drop(ct0_gpu, gpu_idx);
  cuda_drop(out_gpu, gpu_idx);
  cuda_drop(glwe_ct_gpu, gpu_idx);
  cuda_drop(test_vector_idxes_gpu, gpu_idx);

  cuda_destroy_stream(stream, gpu_idx);
}

#endif

void memref_encode_plaintext_with_crt(
    uint64_t *output_allocated, uint64_t *output_aligned,
    uint64_t output_offset, uint64_t output_size, uint64_t output_stride,
    uint64_t input, uint64_t *mods_allocated, uint64_t *mods_aligned,
    uint64_t mods_offset, uint64_t mods_size, uint64_t mods_stride,
    uint64_t mods_product) {

  assert(output_stride == 1 && "Runtime: stride not equal to 1, check "
                               "memref_encode_plaintext_with_crt");

  assert(mods_stride == 1 && "Runtime: stride not equal to 1, check "
                             "memref_encode_plaintext_with_crt");

  for (size_t i = 0; i < (size_t)mods_size; ++i) {
    output_aligned[output_offset + i] =
        encode_crt(input, mods_aligned[mods_offset + i], mods_product);
  }

  return;
}

void memref_encode_expand_lut_for_bootstrap(
    uint64_t *output_lut_allocated, uint64_t *output_lut_aligned,
    uint64_t output_lut_offset, uint64_t output_lut_size,
    uint64_t output_lut_stride, uint64_t *input_lut_allocated,
    uint64_t *input_lut_aligned, uint64_t input_lut_offset,
    uint64_t input_lut_size, uint64_t input_lut_stride, uint32_t poly_size,
    uint32_t out_MESSAGE_BITS) {

  assert(input_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                  "memref_encode_expand_lut_bootstrap");

  assert(output_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                   "memref_encode_expand_lut_bootstrap");

  size_t mega_case_size = output_lut_size / input_lut_size;

  assert((mega_case_size % 2) == 0);

  for (size_t idx = 0; idx < mega_case_size / 2; ++idx) {
    output_lut_aligned[output_lut_offset + idx] =
        input_lut_aligned[input_lut_offset] << (64 - out_MESSAGE_BITS - 1);
  }

  for (size_t idx = (input_lut_size - 1) * mega_case_size + mega_case_size / 2;
       idx < output_lut_size; ++idx) {
    output_lut_aligned[output_lut_offset + idx] =
        -(input_lut_aligned[input_lut_offset] << (64 - out_MESSAGE_BITS - 1));
  }

  for (size_t lut_idx = 1; lut_idx < input_lut_size; ++lut_idx) {
    uint64_t lut_value = input_lut_aligned[input_lut_offset + lut_idx]
                         << (64 - out_MESSAGE_BITS - 1);
    size_t start = mega_case_size * (lut_idx - 1) + mega_case_size / 2;
    for (size_t output_idx = start; output_idx < start + mega_case_size;
         ++output_idx) {
      output_lut_aligned[output_lut_offset + output_idx] = lut_value;
    }
  }

  return;
}

void memref_encode_expand_lut_for_woppbs(
    // Output encoded/expanded lut
    uint64_t *output_lut_allocated, uint64_t *output_lut_aligned,
    uint64_t output_lut_offset, uint64_t output_lut_size,
    uint64_t output_lut_stride,
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
    uint32_t poly_size, uint32_t modulus_product) {

  assert(input_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                  "memref_encode_expand_lut_woppbs");
  assert(output_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                   "memref_encode_expand_lut_woppbs");
  assert(modulus_product > input_lut_size);

  uint64_t lut_crt_size = output_lut_size / crt_decomposition_size;

  for (uint64_t value = 0; value < input_lut_size; value++) {
    uint64_t index_lut = 0;
    uint64_t tmp = 1;

    for (size_t block = 0; block < crt_decomposition_size; block++) {
      auto base = crt_decomposition_aligned[crt_decomposition_offset + block];
      auto bits = crt_bits_aligned[crt_bits_offset + block];
      index_lut += (((value % base) << bits) / base) * tmp;
      tmp <<= bits;
    }

    for (size_t block = 0; block < crt_decomposition_size; block++) {
      auto base = crt_decomposition_aligned[crt_decomposition_offset + block];
      auto v = encode_crt(input_lut_aligned[input_lut_offset + value], base,
                          modulus_product);
      output_lut_aligned[output_lut_offset + block * lut_crt_size + index_lut] =
          v;
    }
  }
}

void memref_add_lwe_ciphertexts_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *ct1_allocated, uint64_t *ct1_aligned,
    uint64_t ct1_offset, uint64_t ct1_size, uint64_t ct1_stride) {
  assert(out_size == ct0_size && out_size == ct1_size &&
         "size of lwe buffer are incompatible");
  size_t lwe_dimension = {out_size - 1};
  CAPI_ASSERT_ERROR(
      default_engine_discard_add_lwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), out_aligned + out_offset,
          ct0_aligned + ct0_offset, ct1_aligned + ct1_offset, lwe_dimension));
}

void memref_add_plaintext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t plaintext) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  size_t lwe_dimension = {out_size - 1};
  CAPI_ASSERT_ERROR(
      default_engine_discard_add_lwe_ciphertext_plaintext_u64_raw_ptr_buffers(
          get_levelled_engine(), out_aligned + out_offset,
          ct0_aligned + ct0_offset, lwe_dimension, plaintext));
}

void memref_mul_cleartext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t cleartext) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  size_t lwe_dimension = {out_size - 1};
  CAPI_ASSERT_ERROR(
      default_engine_discard_mul_lwe_ciphertext_cleartext_u64_raw_ptr_buffers(
          get_levelled_engine(), out_aligned + out_offset,
          ct0_aligned + ct0_offset, lwe_dimension, cleartext));
}

void memref_negate_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  size_t lwe_dimension = {out_size - 1};
  CAPI_ASSERT_ERROR(
      default_engine_discard_opp_lwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), out_aligned + out_offset,
          ct0_aligned + ct0_offset, lwe_dimension));
}

void memref_keyswitch_lwe_u64(uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride,
                              uint32_t level, uint32_t base_log,
                              uint32_t input_lwe_dim, uint32_t output_lwe_dim,
                              mlir::concretelang::RuntimeContext *context) {
  CAPI_ASSERT_ERROR(
      default_engine_discard_keyswitch_lwe_ciphertext_u64_raw_ptr_buffers(
          get_engine(context), get_keyswitch_key_u64(context),
          out_aligned + out_offset, ct0_aligned + ct0_offset));
}

void memref_batched_keyswitch_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint32_t level,
    uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    mlir::concretelang::RuntimeContext *context) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_keyswitch_lwe_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1, level,
        base_log, input_lwe_dim, output_lwe_dim, context);
  }
}

void memref_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context) {

  uint64_t glwe_ct_size = poly_size * (glwe_dim + 1);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size * sizeof(uint64_t));

  CAPI_ASSERT_ERROR(
      default_engine_discard_trivially_encrypt_glwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), glwe_ct, glwe_ct_size,
          tlu_aligned + tlu_offset, poly_size));

  CAPI_ASSERT_ERROR(
      fft_engine_lwe_ciphertext_discarding_bootstrap_u64_raw_ptr_buffers(
          get_fft_engine(context), get_engine(context),
          get_fft_fourier_bootstrap_key_u64(context), out_aligned + out_offset,
          ct0_aligned + ct0_offset, glwe_ct));
  free(glwe_ct);
}

void memref_batched_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size,
    uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t poly_size,
    uint32_t level, uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context) {

  for (size_t i = 0; i < out_size0; i++) {
    memref_bootstrap_lwe_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated, ct0_aligned + i * ct0_size1,
        ct0_offset, ct0_size1, ct0_stride1, tlu_allocated, tlu_aligned,
        tlu_offset, tlu_size, tlu_stride, input_lwe_dim, poly_size, level,
        base_log, glwe_dim, precision, context);
  }
}

uint64_t encode_crt(int64_t plaintext, uint64_t modulus, uint64_t product) {
  return concretelang::clientlib::crt::encode(plaintext, modulus, product);
}

void memref_wop_pbs_crt_buffer(
    // Output 2D memref
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size_0, uint64_t out_size_1, uint64_t out_stride_0,
    uint64_t out_stride_1,
    // Input 2D memref
    uint64_t *in_allocated, uint64_t *in_aligned, uint64_t in_offset,
    uint64_t in_size_0, uint64_t in_size_1, uint64_t in_stride_0,
    uint64_t in_stride_1,
    // clear text lut 1D memref
    uint64_t *lut_ct_allocated, uint64_t *lut_ct_aligned,
    uint64_t lut_ct_offset, uint64_t lut_ct_size, uint64_t lut_ct_stride,
    // CRT decomposition 1D memref
    uint64_t *crt_decomp_allocated, uint64_t *crt_decomp_aligned,
    uint64_t crt_decomp_offset, uint64_t crt_decomp_size,
    uint64_t crt_decomp_stride,
    // Additional crypto parameters
    uint32_t lwe_small_size, uint32_t cbs_level_count, uint32_t cbs_base_log,
    uint32_t polynomial_size,
    // runtime context that hold evluation keys
    mlir::concretelang::RuntimeContext *context) {

  // The compiler should only generates 2D memref<BxS>, where B is the number of
  // ciphertext block and S the lweSize.
  // Check for the strides

  assert(out_stride_1 == 1);
  assert(in_stride_0 == in_size_1 && in_stride_0 == in_size_1);
  // Check for the size B
  assert(out_size_0 == in_size_0 && out_size_0 == crt_decomp_size);
  // Check for the size S
  assert(out_size_1 == in_size_1);

  uint64_t lwe_big_size = in_size_1;

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
      new uint64_t[lwe_small_size * total_number_of_bits_per_block]{0};

  // We make a private copy to apply a subtraction on the body
  auto first_cyphertext = in_aligned + in_offset;
  auto copy_size = crt_decomp_size * lwe_big_size;
  std::vector<uint64_t> in_copy(first_cyphertext, first_cyphertext + copy_size);
  // Extraction of each bit for each block
  for (int64_t i = crt_decomp_size - 1, extract_bits_output_offset = 0; i >= 0;
       extract_bits_output_offset += number_of_bits_per_block[i--]) {
    auto nb_bits_to_extract = number_of_bits_per_block[i];

    auto delta_log = 64 - nb_bits_to_extract;
    auto in_block = &in_copy[lwe_big_size * i];

    // trick ( ct - delta/2 + delta/2^4  )
    uint64_t sub = (uint64_t(1) << (uint64_t(64) - nb_bits_to_extract - 1)) -
                   (uint64_t(1) << (uint64_t(64) - nb_bits_to_extract - 5));
    in_block[lwe_big_size - 1] -= sub;
    CAPI_ASSERT_ERROR(
        fft_engine_lwe_ciphertext_discarding_bit_extraction_unchecked_u64_raw_ptr_buffers(
            context->get_fft_engine(), context->get_default_engine(),
            context->get_fft_fourier_bsk(), context->get_ksk(),
            &extract_bits_output_buffer[lwe_small_size *
                                        extract_bits_output_offset],
            in_block, nb_bits_to_extract, delta_log));
  }

  // Vertical packing
  CAPI_ASSERT_ERROR(
      fft_engine_lwe_ciphertext_vector_discarding_circuit_bootstrap_boolean_vertical_packing_u64_raw_ptr_buffers(
          context->get_fft_engine(), context->get_default_engine(),
          context->get_fft_fourier_bsk(), out_aligned, lwe_big_size,
          crt_decomp_size, extract_bits_output_buffer, lwe_small_size,
          total_number_of_bits_per_block, lut_ct_aligned + lut_ct_offset,
          lut_ct_size, cbs_level_count, cbs_base_log, context->get_fpksk()));
}

void memref_copy_one_rank(uint64_t *src_allocated, uint64_t *src_aligned,
                          uint64_t src_offset, uint64_t src_size,
                          uint64_t src_stride, uint64_t *dst_allocated,
                          uint64_t *dst_aligned, uint64_t dst_offset,
                          uint64_t dst_size, uint64_t dst_stride) {
  assert(src_size == dst_size && "memref_copy_one_rank size differs");
  if (src_stride == dst_stride) {
    memcpy(dst_aligned + dst_offset, src_aligned + src_offset,
           src_size * sizeof(uint64_t));
    return;
  }
  for (size_t i = 0; i < src_size; i++) {
    dst_aligned[dst_offset + i * dst_stride] =
        src_aligned[src_offset + i * src_stride];
  }
}
