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

void encode_and_expand_lut(uint64_t *output, size_t output_size,
                           size_t out_MESSAGE_BITS, const uint64_t *lut,
                           size_t lut_size) {
  assert((output_size % lut_size) == 0);

  size_t mega_case_size = output_size / lut_size;

  assert((mega_case_size % 2) == 0);

  for (size_t idx = 0; idx < mega_case_size / 2; ++idx) {
    output[idx] = lut[0] << (64 - out_MESSAGE_BITS - 1);
  }

  for (size_t idx = (lut_size - 1) * mega_case_size + mega_case_size / 2;
       idx < output_size; ++idx) {
    output[idx] = -(lut[0] << (64 - out_MESSAGE_BITS - 1));
  }

  for (size_t lut_idx = 1; lut_idx < lut_size; ++lut_idx) {
    uint64_t lut_value = lut[lut_idx] << (64 - out_MESSAGE_BITS - 1);
    size_t start = mega_case_size * (lut_idx - 1) + mega_case_size / 2;
    for (size_t output_idx = start; output_idx < start + mega_case_size;
         ++output_idx) {
      output[output_idx] = lut_value;
    }
  }
}

#include "concretelang/ClientLib/CRT.h"
#include "concretelang/Runtime/wrappers.h"

#ifdef CONCRETELANG_CUDA_SUPPORT

// We need to define the double2 struct from the CUDA backend header files
// This shouldn't be defined here, but included along with concrete-cuda header
// files
typedef struct double2 {
  double x, y;
} double2;
// From concrete-cuda
#include "bootstrap.h"
#include "device.h"
#include "keyswitch.h"

void memref_keyswitch_lwe_cuda_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim,
    mlir::concretelang::RuntimeContext *context) {
  // we currently just use the first GPU, but this should be decided
  // dynamically, or during compilation, in the future
  uint32_t gpu_idx = 0;
  uint32_t num_samples = 1;
  void *stream = cuda_create_stream(gpu_idx);
  // move input ciphertext into gpu
  void *ct0_gpu = memcpy_async_ct_to_gpu(ct0_allocated, ct0_aligned, ct0_offset,
                                         ct0_size, ct0_stride, gpu_idx, stream);
  // move output ciphertext into gpu
  void *out_gpu = memcpy_async_ct_to_gpu(out_allocated, out_aligned, out_offset,
                                         out_size, out_stride, gpu_idx, stream);
  void *ksk_gpu = memcpy_async_ksk_to_gpu(context, level, input_lwe_dim,
                                          output_lwe_dim, gpu_idx, stream);
  cuda_keyswitch_lwe_ciphertext_vector_64(stream, out_gpu, ct0_gpu, ksk_gpu,
                                          input_lwe_dim, output_lwe_dim,
                                          base_log, level, num_samples);
  // copy output ciphertext back to cpu
  memcpy_async_ct_to_cpu(out_allocated, out_aligned, out_offset, out_size,
                         out_stride, out_gpu, out_size, gpu_idx, stream);
  cuda_synchronize_device(gpu_idx);
  // free memory that we allocated on gpu
  cuda_drop(ct0_gpu, gpu_idx);
  cuda_drop(out_gpu, gpu_idx);
  cuda_drop(ksk_gpu, gpu_idx);
  cuda_destroy_stream(stream, gpu_idx);
}

void *memcpy_async_ct_to_gpu(uint64_t *ct_allocated, uint64_t *ct_aligned,
                             uint64_t ct_offset, uint64_t ct_size,
                             uint64_t ct_stride, uint32_t gpu_idx,
                             void *stream) {
  size_t buf_size = ct_size * sizeof(uint64_t);
  void *ct_gpu = cuda_malloc(buf_size, gpu_idx);
  cuda_memcpy_async_to_gpu(ct_gpu, ct_aligned + ct_offset, buf_size, stream,
                           gpu_idx);
  return ct_gpu;
}

void *memcpy_async_bsk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t input_lwe_dim, uint32_t poly_size,
                              uint32_t level, uint32_t glwe_dim,
                              uint32_t gpu_idx, void *stream) {
  LweBootstrapKey64 *bsk = get_bootstrap_key_u64(context);
  size_t bsk_buffer_len =
      input_lwe_dim * (glwe_dim + 1) * (glwe_dim + 1) * poly_size * level;
  size_t bsk_buffer_size = bsk_buffer_len * sizeof(uint64_t);
  uint64_t *bsk_buffer =
      (uint64_t *)aligned_alloc(U64_ALIGNMENT, bsk_buffer_size);
  size_t fbsk_gpu_buffer_size = bsk_buffer_len * sizeof(double);
  void *fbsk_gpu = cuda_malloc(fbsk_gpu_buffer_size, gpu_idx);
  CAPI_ASSERT_ERROR(
      default_engine_discard_convert_lwe_bootstrap_key_to_lwe_bootstrap_key_mut_view_u64_raw_ptr_buffers(
          get_levelled_engine(), bsk, bsk_buffer));
  cuda_initialize_twiddles(poly_size, gpu_idx);
  cuda_convert_lwe_bootstrap_key_64(fbsk_gpu, bsk_buffer, stream, gpu_idx,
                                    input_lwe_dim, glwe_dim, level, poly_size);
  // This is currently not 100% async as we have to free CPU memory after
  // conversion
  cuda_synchronize_device(gpu_idx);
  free(bsk_buffer);
  return fbsk_gpu;
}

void *memcpy_async_ksk_to_gpu(mlir::concretelang::RuntimeContext *context,
                              uint32_t level, uint32_t input_lwe_dim,
                              uint32_t output_lwe_dim, uint32_t gpu_idx,
                              void *stream) {
  LweKeyswitchKey64 *ksk = get_keyswitch_key_u64(context);
  size_t ksk_buffer_len = input_lwe_dim * (output_lwe_dim + 1) * level;
  size_t ksk_buffer_size = sizeof(uint64_t) * ksk_buffer_len;
  uint64_t *ksk_buffer =
      (uint64_t *)aligned_alloc(U64_ALIGNMENT, ksk_buffer_size);
  void *ksk_gpu = cuda_malloc(ksk_buffer_size, gpu_idx);
  CAPI_ASSERT_ERROR(
      default_engine_discard_convert_lwe_keyswitch_key_to_lwe_keyswitch_key_mut_view_u64_raw_ptr_buffers(
          get_levelled_engine(), ksk, ksk_buffer));
  cuda_memcpy_async_to_gpu(ksk_gpu, ksk_buffer, ksk_buffer_size, stream,
                           gpu_idx);
  // This is currently not 100% async as we have to free CPU memory after
  // conversion
  cuda_synchronize_device(gpu_idx);
  free(ksk_buffer);
  return ksk_gpu;
}

void memcpy_async_ct_to_cpu(uint64_t *out_allocated, uint64_t *out_aligned,
                            uint64_t out_offset, uint64_t out_size,
                            uint64_t out_stride, void *ct_gpu, size_t size,
                            uint32_t gpu_idx, void *stream) {
  cuda_memcpy_async_to_cpu(out_aligned + out_offset, ct_gpu,
                           size * sizeof(uint64_t), stream, gpu_idx);
}

void free_from_gpu(void *gpu_ptr, uint32_t gpu_idx = 0) {
  cuda_drop(gpu_ptr, gpu_idx);
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
  // we currently just use the first GPU, but this should be decided
  // dynamically, or during compilation, in the future
  uint32_t gpu_idx = 0;
  void *stream = cuda_create_stream(gpu_idx);
  // move bsk to gpu
  void *fbsk_gpu = memcpy_async_bsk_to_gpu(context, input_lwe_dim, poly_size,
                                           level, glwe_dim, gpu_idx, stream);
  // move input ciphertext into gpu
  void *ct0_gpu = memcpy_async_ct_to_gpu(ct0_allocated, ct0_aligned, ct0_offset,
                                         ct0_size, ct0_stride, gpu_idx, stream);
  // move output ciphertext into gpu
  void *out_gpu = memcpy_async_ct_to_gpu(out_allocated, out_aligned, out_offset,
                                         out_size, out_stride, gpu_idx, stream);
  // construct LUT GLWE ciphertext
  uint64_t glwe_ct_len = poly_size * (glwe_dim + 1);
  uint64_t glwe_ct_size = glwe_ct_len * sizeof(uint64_t);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size);
  std::vector<uint64_t> expanded_tabulated_function_array(poly_size);
  encode_and_expand_lut(expanded_tabulated_function_array.data(), poly_size,
                        precision, tlu_aligned + tlu_offset, tlu_size);
  CAPI_ASSERT_ERROR(
      default_engine_discard_trivially_encrypt_glwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), glwe_ct, glwe_ct_len,
          expanded_tabulated_function_array.data(), poly_size));
  // move test vector into gpu
  void *test_vector_gpu =
      cuda_malloc(poly_size * (glwe_dim + 1) * sizeof(uint64_t), gpu_idx);
  cuda_memcpy_async_to_gpu(test_vector_gpu, (void *)glwe_ct, glwe_ct_size,
                           stream, gpu_idx);
  // free LUT ciphertext (CPU)
  free(glwe_ct);
  // move test vector indexes into gpu
  uint32_t num_samples = 1, num_test_vectors = 1, lwe_idx = 0;
  void *test_vector_idxes = malloc(num_samples * sizeof(uint32_t));
  ((uint32_t *)test_vector_idxes)[0] = 0;
  void *test_vector_idxes_gpu =
      cuda_malloc(num_samples * sizeof(uint32_t), gpu_idx);
  cuda_memcpy_async_to_gpu(test_vector_idxes_gpu, test_vector_idxes,
                           num_samples * sizeof(uint32_t), stream, gpu_idx);
  // run gpu bootstrap
  cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
      stream, out_gpu, test_vector_gpu, test_vector_idxes_gpu, ct0_gpu,
      fbsk_gpu, input_lwe_dim, poly_size, base_log, level, num_samples,
      num_test_vectors, lwe_idx, cuda_get_max_shared_memory(gpu_idx));
  // copy output ciphertext back to cpu
  memcpy_async_ct_to_cpu(out_allocated, out_aligned, out_offset, out_size,
                         out_stride, out_gpu, out_size, gpu_idx, stream);
  cuda_synchronize_device(gpu_idx);
  // free memory that we allocated on gpu
  cuda_drop(fbsk_gpu, gpu_idx);
  cuda_drop(ct0_gpu, gpu_idx);
  cuda_drop(out_gpu, gpu_idx);
  cuda_drop(test_vector_gpu, gpu_idx);
  cuda_drop(test_vector_idxes_gpu, gpu_idx);

  cuda_destroy_stream(stream, gpu_idx);
}

#endif

void memref_expand_lut_in_trivial_glwe_ct_u64(
    uint64_t *glwe_ct_allocated, uint64_t *glwe_ct_aligned,
    uint64_t glwe_ct_offset, uint64_t glwe_ct_size, uint64_t glwe_ct_stride,
    uint32_t poly_size, uint32_t glwe_dimension, uint32_t out_precision,
    uint64_t *lut_allocated, uint64_t *lut_aligned, uint64_t lut_offset,
    uint64_t lut_size, uint64_t lut_stride) {

  assert(lut_stride == 1 && "Runtime: stride not equal to 1, check "
                            "memref_expand_lut_in_trivial_glwe_ct_u64");

  assert(glwe_ct_stride == 1 && "Runtime: stride not equal to 1, check "
                                "memref_expand_lut_in_trivial_glwe_ct_u64");

  assert(glwe_ct_size == poly_size * (glwe_dimension + 1));

  std::vector<uint64_t> expanded_tabulated_function_array(poly_size);

  encode_and_expand_lut(expanded_tabulated_function_array.data(), poly_size,
                        out_precision, lut_aligned + lut_offset, lut_size);

  CAPI_ASSERT_ERROR(
      default_engine_discard_trivially_encrypt_glwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), glwe_ct_aligned + glwe_ct_offset, glwe_ct_size,
          expanded_tabulated_function_array.data(), poly_size));

  return;
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

  std::vector<uint64_t> expanded_tabulated_function_array(poly_size);

  encode_and_expand_lut(expanded_tabulated_function_array.data(), poly_size,
                        precision, tlu_aligned + tlu_offset, tlu_size);

  CAPI_ASSERT_ERROR(
      default_engine_discard_trivially_encrypt_glwe_ciphertext_u64_raw_ptr_buffers(
          get_levelled_engine(), glwe_ct, glwe_ct_size,
          expanded_tabulated_function_array.data(), poly_size));

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

void generate_luts_crt_without_padding(
    uint64_t *&luts_crt, uint64_t &total_luts_crt_size, uint64_t *crt_decomp,
    uint64_t *number_of_bits_per_block, size_t crt_size, uint64_t *lut,
    uint64_t lut_size, uint64_t total_number_of_bits, uint64_t modulus,
    uint64_t polynomialSize) {

  uint64_t lut_crt_size = uint64_t(1) << total_number_of_bits;
  lut_crt_size = std::max(lut_crt_size, polynomialSize);

  total_luts_crt_size = crt_size * lut_crt_size;
  luts_crt = (uint64_t *)aligned_alloc(U64_ALIGNMENT,
                                       sizeof(uint64_t) * total_luts_crt_size);

  assert(modulus > lut_size);
  for (uint64_t value = 0; value < lut_size; value++) {
    uint64_t index_lut = 0;

    uint64_t tmp = 1;

    for (size_t block = 0; block < crt_size; block++) {
      auto base = crt_decomp[block];
      auto bits = number_of_bits_per_block[block];
      index_lut += (((value % base) << bits) / base) * tmp;
      tmp <<= bits;
    }

    for (size_t block = 0; block < crt_size; block++) {
      auto base = crt_decomp[block];
      auto v = encode_crt(lut[value], base, modulus);
      luts_crt[block * lut_crt_size + index_lut] = v;
    }
  }
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
  uint64_t message_modulus = 1;
  auto number_of_bits_per_block = new uint64_t[crt_decomp_size]();
  for (uint64_t i = 0; i < crt_decomp_size; i++) {
    uint64_t modulus = crt_decomp_aligned[i + crt_decomp_offset];
    uint64_t nb_bit_to_extract =
        static_cast<uint64_t>(ceil(log2(static_cast<double>(modulus))));
    number_of_bits_per_block[i] = nb_bit_to_extract;

    total_number_of_bits_per_block += nb_bit_to_extract;
    message_modulus *= modulus;
  }

  // Create the buffer of ciphertexts for storing the total number of bits to
  // extract.
  // The extracted bit should be in the following order:
  //
  // [msb(m%crt[n-1])..lsb(m%crt[n-1])...msb(m%crt[0])..lsb(m%crt[0])] where n
  // is the size of the crt decomposition
  auto extract_bits_output_buffer =
      new uint64_t[lwe_small_size * total_number_of_bits_per_block]{0};

  // Extraction of each bit for each block
  for (int64_t i = crt_decomp_size - 1, extract_bits_output_offset = 0; i >= 0;
       extract_bits_output_offset += number_of_bits_per_block[i--]) {
    auto nb_bits_to_extract = number_of_bits_per_block[i];

    auto delta_log = 64 - nb_bits_to_extract;
    auto in_block = &in_aligned[lwe_big_size * i];

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

  uint64_t *luts_crt;
  uint64_t luts_crt_size;
  generate_luts_crt_without_padding(
      luts_crt, luts_crt_size, crt_decomp_aligned, number_of_bits_per_block,
      crt_decomp_size, lut_ct_aligned, lut_ct_size,
      total_number_of_bits_per_block, message_modulus, polynomial_size);

  // Vertical packing
  CAPI_ASSERT_ERROR(
      fft_engine_lwe_ciphertext_vector_discarding_circuit_bootstrap_boolean_vertical_packing_u64_raw_ptr_buffers(
          context->get_fft_engine(), context->get_default_engine(),
          context->get_fft_fourier_bsk(), out_aligned, lwe_big_size,
          crt_decomp_size, extract_bits_output_buffer, lwe_small_size,
          total_number_of_bits_per_block, luts_crt, luts_crt_size,
          cbs_level_count, cbs_base_log, context->get_fpksk()));

  free(luts_crt);
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
