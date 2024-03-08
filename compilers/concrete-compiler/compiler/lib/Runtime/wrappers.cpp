// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/wrappers.h"
#include "concrete-cpu.h"
#include "concretelang/Common/Error.h"
#include <assert.h>
#include <bitset>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "concretelang/Common/CRT.h"
#include "concretelang/Runtime/wrappers.h"

#ifdef CONCRETELANG_CUDA_SUPPORT

// CUDA memory utils function /////////////////////////////////////////////////

void *memcpy_async_bsk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t input_lwe_dim, uint32_t poly_size,
                              uint32_t level, uint32_t glwe_dim,
                              uint32_t gpu_idx, void *stream,
                              uint32_t bsk_idx) {
  return context->get_bsk_gpu(input_lwe_dim, poly_size, level, glwe_dim,
                              gpu_idx, stream, bsk_idx);
}

void *memcpy_async_ksk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t level, uint32_t input_lwe_dim,
                              uint32_t output_lwe_dim, uint32_t gpu_idx,
                              void *stream, uint32_t ksk_idx) {
  return context->get_ksk_gpu(level, input_lwe_dim, output_lwe_dim, gpu_idx,
                              stream, ksk_idx);
}

void *alloc_and_memcpy_async_to_gpu(uint64_t *buf_ptr, uint64_t buf_offset,
                                    uint64_t buf_size, uint32_t gpu_idx,
                                    void *stream) {
  size_t buf_size_ = buf_size * sizeof(uint64_t);
  void *ct_gpu = cuda_malloc_async(buf_size_, (cudaStream_t *)stream, gpu_idx);
  cuda_memcpy_async_to_gpu(ct_gpu, buf_ptr + buf_offset, buf_size_,
                           (cudaStream_t *)stream, gpu_idx);
  return ct_gpu;
}

void memcpy_async_to_cpu(uint64_t *buf_ptr, uint64_t buf_offset,
                         uint64_t buf_size, void *buf_gpu, uint32_t gpu_idx,
                         void *stream) {
  cuda_memcpy_async_to_cpu(buf_ptr + buf_offset, buf_gpu,
                           buf_size * sizeof(uint64_t), (cudaStream_t *)stream,
                           gpu_idx);
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
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t ksk_index,
    mlir::concretelang::RuntimeContext *context) {
  assert(out_stride == 1);
  assert(ct0_stride == 1);
  memref_batched_keyswitch_lwe_cuda_u64(
      // Output 1D memref as 2D memref
      out_allocated, out_aligned, out_offset, 1, out_size, out_size, out_stride,
      // Output 1D memref as 2D memref
      ct0_allocated, ct0_aligned, ct0_offset, 1, ct0_size, ct0_size, ct0_stride,
      // Keyswitch additional arguments
      level, base_log, input_lwe_dim, output_lwe_dim, ksk_index, context);
}

void memref_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context) {
  memref_batched_bootstrap_lwe_cuda_u64(
      // Output 1D memref as 2D memref
      out_allocated, out_aligned, out_offset, 1, out_size, out_size, out_stride,
      // Input 1D memref as 2D memref
      ct0_allocated, ct0_aligned, ct0_offset, 1, ct0_size, ct0_size, ct0_stride,
      // Table lookup memref
      tlu_allocated, tlu_aligned, tlu_offset, tlu_size, tlu_stride,
      // Bootstrap additional arguments
      input_lwe_dim, poly_size, level, base_log, glwe_dim, bsk_index, context);
}

// Batched CUDA function //////////////////////////////////////////////////////

void memref_batched_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint32_t level,
    uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    uint32_t ksk_index, mlir::concretelang::RuntimeContext *context) {
  assert(ksk_index == 0 && "multiple ksk is not yet implemented on GPU");
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
  void *ksk_gpu =
      memcpy_async_ksk_to_gpu(context, level, input_lwe_dim, output_lwe_dim,
                              gpu_idx, stream, ksk_index);
  // Move the input and output batch of ciphertexts to the GPU
  // TODO: The allocation should be done by the compiler codegen
  void *ct0_gpu = alloc_and_memcpy_async_to_gpu(
      ct0_aligned, ct0_offset, ct0_batch_size, gpu_idx, (cudaStream_t *)stream);
  void *out_gpu = cuda_malloc_async(out_batch_size * sizeof(uint64_t),
                                    (cudaStream_t *)stream, gpu_idx);
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
  cuda_destroy_stream((cudaStream_t *)stream, gpu_idx);
}

void memref_batched_bootstrap_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size,
    uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t poly_size,
    uint32_t level, uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context) {
  assert(bsk_index == 0 && "multiple bsk is not yet implemented on GPU");
  assert(out_size0 == ct0_size0);
  assert(out_size1 == glwe_dim * poly_size + 1);
  // TODO: Multi GPU
  uint32_t gpu_idx = 0;
  uint32_t num_samples = out_size0;
  uint64_t ct0_batch_size = ct0_size0 * ct0_size1;
  uint64_t out_batch_size = out_size0 * out_size1;
  int8_t *pbs_buffer = nullptr;

  // Create the cuda stream
  // TODO: Should be created by the compiler codegen
  void *stream = cuda_create_stream(gpu_idx);
  // Get the pointer on the bootstraping key on the GPU
  void *fbsk_gpu =
      memcpy_async_bsk_to_gpu(context, input_lwe_dim, poly_size, level,
                              glwe_dim, gpu_idx, stream, bsk_index);
  // Move the input and output batch of ciphertext to the GPU
  // TODO: The allocation should be done by the compiler codegen
  void *ct0_gpu = alloc_and_memcpy_async_to_gpu(
      ct0_aligned, ct0_offset, ct0_batch_size, gpu_idx, (cudaStream_t *)stream);
  void *out_gpu = cuda_malloc_async(out_batch_size * sizeof(uint64_t),
                                    (cudaStream_t *)stream, gpu_idx);
  // Construct the glwe accumulator (on CPU)
  // TODO: Should be done outside of the bootstrap call, compile time if
  // possible. Refactor in progress
  uint64_t glwe_ct_size = poly_size * (glwe_dim + 1);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size * sizeof(uint64_t));
  auto tlu = tlu_aligned + tlu_offset;

  // Glwe trivial encryption
  for (size_t i = 0; i < poly_size * glwe_dim; i++) {
    glwe_ct[i] = 0;
  }
  for (size_t i = 0; i < poly_size; i++) {
    glwe_ct[poly_size * glwe_dim + i] = tlu[i];
  }

  // Move the glwe accumulator to the GPU
  void *glwe_ct_gpu = alloc_and_memcpy_async_to_gpu(
      glwe_ct, 0, glwe_ct_size, gpu_idx, (cudaStream_t *)stream);

  // Move test vector indexes to the GPU, the test vector indexes is set of 0
  uint32_t num_test_vectors = 1, lwe_idx = 0,
           test_vector_idxes_size = num_samples * sizeof(uint64_t);
  void *test_vector_idxes = malloc(test_vector_idxes_size);
  memset(test_vector_idxes, 0, test_vector_idxes_size);
  void *test_vector_idxes_gpu = cuda_malloc_async(
      test_vector_idxes_size, (cudaStream_t *)stream, gpu_idx);
  cuda_memcpy_async_to_gpu(test_vector_idxes_gpu, test_vector_idxes,
                           test_vector_idxes_size, (cudaStream_t *)stream,
                           gpu_idx);
  // Allocate PBS buffer on GPU
  scratch_cuda_bootstrap_amortized_64(
      stream, gpu_idx, &pbs_buffer, glwe_dim, poly_size, num_samples,
      cuda_get_max_shared_memory(gpu_idx), true);
  // Run the bootstrap kernel on the GPU
  cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
      stream, gpu_idx, out_gpu, glwe_ct_gpu, test_vector_idxes_gpu, ct0_gpu,
      fbsk_gpu, pbs_buffer, input_lwe_dim, glwe_dim, poly_size, base_log, level,
      num_samples, num_test_vectors, lwe_idx,
      cuda_get_max_shared_memory(gpu_idx));
  cleanup_cuda_bootstrap_amortized(stream, gpu_idx, &pbs_buffer);
  // Copy the output batch of ciphertext back to CPU
  memcpy_async_to_cpu(out_aligned, out_offset, out_batch_size, out_gpu, gpu_idx,
                      stream);
  // free memory that we allocated on gpu
  cuda_drop_async(ct0_gpu, (cudaStream_t *)stream, gpu_idx);
  cuda_drop_async(out_gpu, (cudaStream_t *)stream, gpu_idx);
  cuda_drop_async(glwe_ct_gpu, (cudaStream_t *)stream, gpu_idx);
  cuda_drop_async(test_vector_idxes_gpu, (cudaStream_t *)stream, gpu_idx);
  cudaStreamSynchronize(*(cudaStream_t *)stream);
  // Free the glwe accumulator (on CPU)
  free(glwe_ct);
  cuda_destroy_stream((cudaStream_t *)stream, gpu_idx);
}

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
    mlir::concretelang::RuntimeContext *context) {
  assert(bsk_index == 0 && "multiple bsk is not yet implemented on GPU");
  assert(out_size0 == ct0_size0);
  assert(out_size1 == glwe_dim * poly_size + 1);
  assert((out_size0 == tlu_size0 || tlu_size0 == 1) &&
         "Number of LUTs does not match batch size");
  // TODO: Multi GPU
  uint32_t gpu_idx = 0;
  uint32_t num_samples = out_size0;
  uint32_t num_lut_vectors = tlu_size0;
  uint64_t ct0_batch_size = ct0_size0 * ct0_size1;
  uint64_t out_batch_size = out_size0 * out_size1;
  int8_t *pbs_buffer = nullptr;

  // Create the cuda stream
  // TODO: Should be created by the compiler codegen
  void *stream = cuda_create_stream(gpu_idx);
  // Get the pointer on the bootstraping key on the GPU
  void *fbsk_gpu =
      memcpy_async_bsk_to_gpu(context, input_lwe_dim, poly_size, level,
                              glwe_dim, gpu_idx, stream, bsk_index);
  // Move the input and output batch of ciphertext to the GPU
  // TODO: The allocation should be done by the compiler codegen
  void *ct0_gpu = alloc_and_memcpy_async_to_gpu(
      ct0_aligned, ct0_offset, ct0_batch_size, gpu_idx, (cudaStream_t *)stream);
  void *out_gpu = cuda_malloc_async(out_batch_size * sizeof(uint64_t),
                                    (cudaStream_t *)stream, gpu_idx);
  // Construct the glwe accumulator (on CPU)
  // TODO: Should be done outside of the bootstrap call, compile time if
  // possible. Refactor in progress
  uint64_t glwe_ct_size = poly_size * (glwe_dim + 1) * num_lut_vectors;
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size * sizeof(uint64_t));
  auto tlu = tlu_aligned + tlu_offset;

  // Glwe trivial encryption
  size_t pos = 0, postlu = 0;
  for (size_t l = 0; l < num_lut_vectors; ++l) {
    for (size_t i = 0; i < poly_size * glwe_dim; i++) {
      glwe_ct[pos++] = 0;
    }
    for (size_t i = 0; i < poly_size; i++) {
      glwe_ct[pos++] = tlu[postlu++];
    }
  }

  // Move the glwe accumulator to the GPU
  void *glwe_ct_gpu = alloc_and_memcpy_async_to_gpu(
      glwe_ct, 0, glwe_ct_size, gpu_idx, (cudaStream_t *)stream);

  // Move test vector indexes to the GPU, the test vector indexes is set of 0
  uint32_t lwe_idx = 0, test_vector_idxes_size = num_samples * sizeof(uint64_t);
  uint64_t *test_vector_idxes = (uint64_t *)malloc(test_vector_idxes_size);
  if (num_lut_vectors == 1) {
    memset((void *)test_vector_idxes, 0, test_vector_idxes_size);
  } else {
    assert(num_lut_vectors == num_samples);
    for (size_t i = 0; i < num_lut_vectors; ++i)
      test_vector_idxes[i] = i;
  }
  void *test_vector_idxes_gpu = cuda_malloc_async(
      test_vector_idxes_size, (cudaStream_t *)stream, gpu_idx);
  cuda_memcpy_async_to_gpu(test_vector_idxes_gpu, (void *)test_vector_idxes,
                           test_vector_idxes_size, (cudaStream_t *)stream,
                           gpu_idx);
  // Allocate PBS buffer on GPU
  scratch_cuda_bootstrap_amortized_64(
      stream, gpu_idx, &pbs_buffer, glwe_dim, poly_size, num_samples,
      cuda_get_max_shared_memory(gpu_idx), true);
  // Run the bootstrap kernel on the GPU
  cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
      stream, gpu_idx, out_gpu, glwe_ct_gpu, test_vector_idxes_gpu, ct0_gpu,
      fbsk_gpu, pbs_buffer, input_lwe_dim, glwe_dim, poly_size, base_log, level,
      num_samples, num_lut_vectors, lwe_idx,
      cuda_get_max_shared_memory(gpu_idx));
  cleanup_cuda_bootstrap_amortized(stream, gpu_idx, &pbs_buffer);
  // Copy the output batch of ciphertext back to CPU
  memcpy_async_to_cpu(out_aligned, out_offset, out_batch_size, out_gpu, gpu_idx,
                      stream);
  // free memory that we allocated on gpu
  cuda_drop_async(ct0_gpu, (cudaStream_t *)stream, gpu_idx);
  cuda_drop_async(out_gpu, (cudaStream_t *)stream, gpu_idx);
  cuda_drop_async(glwe_ct_gpu, (cudaStream_t *)stream, gpu_idx);
  cuda_drop_async(test_vector_idxes_gpu, (cudaStream_t *)stream, gpu_idx);
  cudaStreamSynchronize(*(cudaStream_t *)stream);
  // Free the glwe accumulator (on CPU)
  free(glwe_ct);
  cuda_destroy_stream((cudaStream_t *)stream, gpu_idx);
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
    uint32_t out_MESSAGE_BITS, bool is_signed) {

  assert(input_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                  "memref_encode_expand_lut_bootstrap");

  assert(output_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                   "memref_encode_expand_lut_bootstrap");

  size_t mega_case_size = output_lut_size / input_lut_size;

  assert((mega_case_size % 2) == 0);

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
  }
  for (size_t idx = (input_lut_size - 1) * mega_case_size + mega_case_size / 2;
       idx < output_lut_size; ++idx) {
    output_lut_aligned[output_lut_offset + idx] =
        -(input_lut_aligned[input_lut_offset + indexMap(0)]
          << (64 - out_MESSAGE_BITS - 1));
  }

  // Treats the other ut values.
  for (size_t lut_idx = 1; lut_idx < input_lut_size; ++lut_idx) {
    uint64_t lut_value = input_lut_aligned[input_lut_offset + indexMap(lut_idx)]
                         << (64 - out_MESSAGE_BITS - 1);
    size_t start = mega_case_size * (lut_idx - 1) + mega_case_size / 2;
    for (size_t output_idx = start; output_idx < start + mega_case_size;
         ++output_idx) {
      output_lut_aligned[output_lut_offset + output_idx] = lut_value;
    }
  }

  return;
}

void memref_encode_lut_for_crt_woppbs(
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

  assert(input_lut_stride == 1 && "Runtime: stride not equal to 1, check "
                                  "memref_encode_lut_woppbs");
  assert(output_lut_stride0 == output_lut_size1 &&
         "Runtime: out dim stride not equal to in_dim size, check "
         "memref_encode_lut_woppbs");
  assert(output_lut_stride1 == 1 && "Runtime: stride not equal to 1, check "
                                    "memref_encode_lut_woppbs");

  assert(modulus_product >= input_lut_size);

  // Initialize lut cases not supposed to be reached
  for (uint64_t i = 0; i < output_lut_size0 * output_lut_size1; i++) {
    output_lut_aligned[output_lut_offset + i] = 0;
  }

  // When the woppbs is executed on encrypted signed integers, the index of the
  // lut elements must be adapted to fit the way signed are encrypted in CRT
  // (to ensure the lookup falls into the proper case).
  // This map takes care about properly indexing into the output lut depending
  // on what bootstrap gets executed.
  std::function<uint64_t(uint64_t)> indexMap;
  if (!is_signed) {
    // When not signed, the integer values are encoded in increasing order. That
    // is (example of 9 bits values, using crt decomposition [5,7,16]):
    //
    // |0     511|
    // |---------|
    // |0     511|
    //
    // is encoded as
    //
    // |0   511|  INVALID  |
    // |-------|-----------|
    // |0   511|512     559|
    //
    // Where on top are represented the semantic values, and below, the actual
    // encoding of values, either on uint64_t or as increasing crt values.
    //
    // As a consequence, there is nothing particular to do to map the index of
    // the input lut to an index of the output lut.
    indexMap = [=](uint64_t plaintext) { return plaintext; };
  } else {
    // When signed, the integer values are encoded in a way that resembles 2s
    // complement. That is (example of 9 bits values, using crt decomposition
    // [5,7,16]):
    //
    // |0     255|-256    -1|
    // |---------|----------|
    // |0     255|256    511|
    //
    // is encoded as
    //
    // |0     255|   INVALID   |-256    -1|
    // |---------|-------------|----------|
    // |0     255|256       303|304    559|
    //
    // Where on top are represented the semantic values, and below, the actual
    // encoding of values, either on uint64_t or as increasing crt values.
    //
    // As a consequence, to map the index of the input lut to an index of the
    // output lut we must take care of crossing the invalid range in between
    // positive values and negative values.
    indexMap = [=](uint64_t plaintext) {
      if (plaintext >= (input_lut_size / 2)) {
        plaintext += modulus_product - input_lut_size;
      }
      return plaintext;
    };
  }

  uint64_t log_lut_crt_size = 0;

  for (size_t in_block = 0; in_block < crt_decomposition_size; in_block++) {
    auto bits_count = crt_bits_aligned[crt_bits_offset + in_block];
    log_lut_crt_size += bits_count;
  }

  uint64_t lut_crt_size = 1 << log_lut_crt_size;
  assert(lut_crt_size == output_lut_size1);
  assert(crt_decomposition_size == output_lut_size0);

  for (uint64_t in_index = 0; in_index < input_lut_size; in_index++) {
    uint64_t out_index = 0;

    {
      uint64_t total_bit_count = 0;
      for (size_t in_block = 0; in_block < crt_decomposition_size; in_block++) {
        auto in_base =
            crt_decomposition_aligned[crt_decomposition_offset + in_block];
        auto bits_count = crt_bits_aligned[crt_bits_offset + in_block];
        out_index += (((indexMap(in_index) % in_base) << bits_count) / in_base)
                     << total_bit_count;
        total_bit_count += bits_count;
      }
    }

    for (size_t out_block = 0; out_block < crt_decomposition_size;
         out_block++) {
      auto out_base =
          crt_decomposition_aligned[crt_decomposition_offset + out_block];
      auto v = encode_crt(input_lut_aligned[input_lut_offset + in_index],
                          out_base, modulus_product);
      output_lut_aligned[output_lut_offset + out_block * lut_crt_size +
                         out_index] = v;
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
  size_t lwe_dimension = out_size - 1;
  concrete_cpu_add_lwe_ciphertext_u64(out_aligned + out_offset,
                                      ct0_aligned + ct0_offset,
                                      ct1_aligned + ct1_offset, lwe_dimension);
}

void memref_add_plaintext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t plaintext) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  size_t lwe_dimension = out_size - 1;
  concrete_cpu_add_plaintext_lwe_ciphertext_u64(out_aligned + out_offset,
                                                ct0_aligned + ct0_offset,
                                                plaintext, lwe_dimension);
}

void memref_mul_cleartext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t cleartext) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  size_t lwe_dimension = out_size - 1;
  concrete_cpu_mul_cleartext_lwe_ciphertext_u64(out_aligned + out_offset,
                                                ct0_aligned + ct0_offset,
                                                cleartext, lwe_dimension);
}

void memref_negate_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  size_t lwe_dimension = {out_size - 1};
  concrete_cpu_negate_lwe_ciphertext_u64(
      out_aligned + out_offset, ct0_aligned + ct0_offset, lwe_dimension);
}

void memref_keyswitch_lwe_u64(uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride,
                              uint32_t decomposition_level_count,
                              uint32_t decomposition_base_log,
                              uint32_t input_dimension,
                              uint32_t output_dimension, uint32_t ksk_index,
                              mlir::concretelang::RuntimeContext *context) {
  assert(out_stride == 1 && ct0_stride == 1);
  // Get keyswitch key
  const uint64_t *keyswitch_key = context->keyswitch_key_buffer(ksk_index);
  // Get stack parameter
  concrete_cpu_keyswitch_lwe_ciphertext_u64(
      out_aligned + out_offset, ct0_aligned + ct0_offset, keyswitch_key,
      decomposition_level_count, decomposition_base_log, input_dimension,
      output_dimension);
}

void memref_batched_add_lwe_ciphertexts_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *ct1_allocated,
    uint64_t *ct1_aligned, uint64_t ct1_offset, uint64_t ct1_size0,
    uint64_t ct1_size1, uint64_t ct1_stride0, uint64_t ct1_stride1) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_add_lwe_ciphertexts_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1,
        ct1_allocated + i * ct1_size1, ct1_aligned + i * ct1_size1, ct1_offset,
        ct1_size1, ct1_stride1);
  }
}

void memref_batched_add_plaintext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *ct1_allocated,
    uint64_t *ct1_aligned, uint64_t ct1_offset, uint64_t ct1_size,
    uint64_t ct1_stride) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_add_plaintext_lwe_ciphertext_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1,
        *(ct1_aligned + ct1_offset + i * ct1_stride));
  }
}

void memref_batched_add_plaintext_cst_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t plaintext) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_add_plaintext_lwe_ciphertext_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1,
        plaintext);
  }
}

void memref_batched_mul_cleartext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *ct1_allocated,
    uint64_t *ct1_aligned, uint64_t ct1_offset, uint64_t ct1_size,
    uint64_t ct1_stride) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_mul_cleartext_lwe_ciphertext_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1,
        *(ct1_aligned + ct1_offset + i * ct1_stride));
  }
}

void memref_batched_mul_cleartext_cst_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t cleartext) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_mul_cleartext_lwe_ciphertext_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1,
        cleartext);
  }
}

void memref_batched_negate_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_negate_lwe_ciphertext_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1);
  }
}

void memref_batched_keyswitch_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint32_t level,
    uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    uint32_t ksk_index, mlir::concretelang::RuntimeContext *context) {
  for (size_t i = 0; i < ct0_size0; i++) {
    memref_keyswitch_lwe_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated + i * ct0_size1,
        ct0_aligned + i * ct0_size1, ct0_offset, ct0_size1, ct0_stride1, level,
        base_log, input_lwe_dim, output_lwe_dim, ksk_index, context);
  }
}

void memref_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dimension, uint32_t polynomial_size,
    uint32_t decomposition_level_count, uint32_t decomposition_base_log,
    uint32_t glwe_dimension, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context) {

  uint64_t glwe_ct_size = polynomial_size * (glwe_dimension + 1);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size * sizeof(uint64_t));
  auto tlu = tlu_aligned + tlu_offset;

  // Glwe trivial encryption
  for (size_t i = 0; i < polynomial_size * glwe_dimension; i++) {
    glwe_ct[i] = 0;
  }
  for (size_t i = 0; i < polynomial_size; i++) {
    glwe_ct[polynomial_size * glwe_dimension + i] = tlu[i];
  }

  // Get fourrier bootstrap key
  const auto &fft = context->fft(bsk_index);
  auto bootstrap_key = context->fourier_bootstrap_key_buffer(bsk_index);
  // Get stack parameter
  size_t scratch_size;
  size_t scratch_align;
  concrete_cpu_bootstrap_lwe_ciphertext_u64_scratch(
      &scratch_size, &scratch_align, glwe_dimension, polynomial_size, fft);
  // Allocate scratch
  auto scratch = (uint8_t *)aligned_alloc(scratch_align, scratch_size);

  // Bootstrap
  concrete_cpu_bootstrap_lwe_ciphertext_u64(
      out_aligned + out_offset, ct0_aligned + ct0_offset, glwe_ct,
      bootstrap_key, decomposition_level_count, decomposition_base_log,
      glwe_dimension, polynomial_size, input_lwe_dimension, fft, scratch,
      scratch_size);

  free(glwe_ct);
  free(scratch);
}

void memref_batched_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size0, uint64_t out_size1, uint64_t out_stride0,
    uint64_t out_stride1, uint64_t *ct0_allocated, uint64_t *ct0_aligned,
    uint64_t ct0_offset, uint64_t ct0_size0, uint64_t ct0_size1,
    uint64_t ct0_stride0, uint64_t ct0_stride1, uint64_t *tlu_allocated,
    uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t tlu_size,
    uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t poly_size,
    uint32_t level, uint32_t base_log, uint32_t glwe_dim, uint32_t bsk_index,
    mlir::concretelang::RuntimeContext *context) {

  for (size_t i = 0; i < out_size0; i++) {
    memref_bootstrap_lwe_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated, ct0_aligned + i * ct0_size1,
        ct0_offset, ct0_size1, ct0_stride1, tlu_allocated, tlu_aligned,
        tlu_offset, tlu_size, tlu_stride, input_lwe_dim, poly_size, level,
        base_log, glwe_dim, bsk_index, context);
  }
}

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
    mlir::concretelang::RuntimeContext *context) {
  assert(out_size0 == tlu_size0 && "Number of LUTs does not match batch size");
  for (size_t i = 0; i < out_size0; i++) {
    memref_bootstrap_lwe_u64(
        out_allocated + i * out_size1, out_aligned + i * out_size1, out_offset,
        out_size1, out_stride1, ct0_allocated, ct0_aligned + i * ct0_size1,
        ct0_offset, ct0_size1, ct0_stride1, tlu_allocated,
        tlu_aligned + i * tlu_size1, tlu_offset, tlu_size1, tlu_stride1,
        input_lwe_dim, poly_size, level, base_log, glwe_dim, bsk_index,
        context);
  }
}

uint64_t encode_crt(int64_t plaintext, uint64_t modulus, uint64_t product) {
  return concretelang::crt::encode(plaintext, modulus, product);
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
    uint32_t polynomial_size,
    // Key Indices,
    uint32_t ksk_index, uint32_t bsk_index, uint32_t pksk_index,
    // runtime context that hold evaluation keys
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

  uint64_t lwe_small_size = lwe_small_dim + 1;

  assert(out_size_1 == in_size_1);
  uint64_t lwe_big_size = in_size_1;
  uint64_t lwe_big_dim = lwe_big_size - 1;
  assert(lwe_big_dim % polynomial_size == 0);

  assert(lwe_big_dim % polynomial_size == 0);
  uint64_t glwe_dim = lwe_big_dim / polynomial_size;

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
  auto first_ciphertext = in_aligned + in_offset;
  auto copy_size = crt_decomp_size * lwe_big_size;
  std::vector<uint64_t> in_copy(first_ciphertext, first_ciphertext + copy_size);
  // Extraction of each bit for each block

  const auto &fft = context->fft(bsk_index);
  auto bootstrap_key = context->fourier_bootstrap_key_buffer(bsk_index);
  auto keyswicth_key = context->keyswitch_key_buffer(ksk_index);

  for (int64_t i = crt_decomp_size - 1, extract_bits_output_offset = 0; i >= 0;
       extract_bits_output_offset += number_of_bits_per_block[i--]) {
    auto nb_bits_to_extract = number_of_bits_per_block[i];

    size_t delta_log = 64 - nb_bits_to_extract;

    auto in_block = &in_copy[lwe_big_size * i];

    // trick ( ct - delta/2 + delta/2^4  )
    uint64_t sub = (uint64_t(1) << (uint64_t(64) - nb_bits_to_extract - 1)) -
                   (uint64_t(1) << (uint64_t(64) - nb_bits_to_extract - 5));
    in_block[lwe_big_size - 1] -= sub;

    size_t scratch_size;
    size_t scratch_align;
    concrete_cpu_extract_bit_lwe_ciphertext_u64_scratch(
        &scratch_size, &scratch_align, lwe_small_dim, lwe_big_dim, glwe_dim,
        polynomial_size, fft);
    // Allocate scratch
    auto *scratch = (uint8_t *)aligned_alloc(scratch_align, scratch_size);

    concrete_cpu_extract_bit_lwe_ciphertext_u64(
        &extract_bits_output_buffer[lwe_small_size *
                                    extract_bits_output_offset],
        in_block, bootstrap_key, keyswicth_key, lwe_small_dim,
        nb_bits_to_extract, lwe_big_dim, nb_bits_to_extract, delta_log,
        bsk_level_count, bsk_base_log, glwe_dim, polynomial_size, lwe_small_dim,
        ksk_level_count, ksk_base_log, lwe_big_dim, lwe_small_dim, fft, scratch,
        scratch_size);

    free(scratch);
  }

  size_t ct_in_count = total_number_of_bits_per_block;
  size_t lut_size = 1 << ct_in_count;
  size_t ct_out_count = out_size_0;
  size_t lut_count = ct_out_count;

  assert(lut_ct_size0 == lut_count);
  assert(lut_ct_size1 == lut_size);

  // Vertical packing
  size_t scratch_size;
  size_t scratch_align;
  concrete_cpu_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64_scratch(
      &scratch_size, &scratch_align, ct_out_count, lwe_small_dim, ct_in_count,
      lut_size, lut_count, glwe_dim, polynomial_size, polynomial_size,
      cbs_level_count, fft);

  auto *scratch = (uint8_t *)aligned_alloc(scratch_align, scratch_size);

  auto fp_keyswicth_key = context->fp_keyswitch_key_buffer(pksk_index);

  concrete_cpu_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64(
      out_aligned + out_offset, extract_bits_output_buffer,
      lut_ct_aligned + lut_ct_offset, bootstrap_key, fp_keyswicth_key,
      lwe_big_dim, ct_out_count, lwe_small_dim, ct_in_count, lut_size,
      lut_count, bsk_level_count, bsk_base_log, glwe_dim, polynomial_size,
      lwe_small_dim, fpksk_level_count, fpksk_base_log, lwe_big_dim, glwe_dim,
      polynomial_size, glwe_dim + 1, cbs_level_count, cbs_base_log, fft,
      scratch, scratch_size);

  free(scratch);
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

void memref_trace_ciphertext(uint64_t *ct0_allocated, uint64_t *ct0_aligned,
                             uint64_t ct0_offset, uint64_t ct0_size,
                             uint64_t ct0_stride, char *message_ptr,
                             uint32_t message_len, uint32_t msb) {
  std::string message{message_ptr, (size_t)message_len};
  std::cout << message << " : ";
  std::bitset<64> bits{ct0_aligned[ct0_offset + ct0_size - 1]};
  std::string bitstring = bits.to_string();
  bitstring.insert(msb, 1, ' ');
  std::cout << bitstring << std::endl;
}

void memref_trace_plaintext(uint64_t input, uint64_t input_width,
                            char *message_ptr, uint32_t message_len,
                            uint32_t msb) {
  std::string message{message_ptr, (size_t)message_len};
  std::cout << message << " : ";
  std::bitset<64> bits{input};
  std::string bitstring = bits.to_string();
  bitstring.erase(0, 64 - input_width);
  bitstring.insert(msb, 1, ' ');
  std::cout << bitstring << std::endl;
}

void memref_trace_message(char *message_ptr, uint32_t message_len) {
  std::string message{message_ptr, (size_t)message_len};
  std::cout << message << std::flush;
}
