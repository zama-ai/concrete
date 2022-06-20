// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/wrappers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/seeder.h"
#include <assert.h>
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
// This helper function expands the input LUT into output, duplicating values as
// needed to fill mega cases, taking care of the encoding and the half mega case
// shift in the process as well. All sizes should be powers of 2.
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
                              mlir::concretelang::RuntimeContext *context) {
  CAPI_ASSERT_ERROR(
      default_engine_discard_keyswitch_lwe_ciphertext_u64_raw_ptr_buffers(
          get_engine(context), get_keyswitch_key_u64(context),
          out_aligned + out_offset, ct0_aligned + ct0_offset));
}

void memref_bootstrap_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *glwe_ct_allocated, uint64_t *glwe_ct_aligned,
    uint64_t glwe_ct_offset, uint64_t glwe_ct_size, uint64_t glwe_ct_stride,
    mlir::concretelang::RuntimeContext *context) {
  CAPI_ASSERT_ERROR(
      fftw_engine_lwe_ciphertext_discarding_bootstrap_u64_raw_ptr_buffers(
          get_fftw_engine(context), get_engine(context),
          get_bootstrap_key_u64(context), out_aligned + out_offset,
          ct0_aligned + ct0_offset, glwe_ct_aligned + glwe_ct_offset));
}

uint64_t encode_crt(int64_t plaintext, uint64_t modulus, uint64_t product) {
  return concretelang::clientlib::crt::encode(plaintext, modulus, product);
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
