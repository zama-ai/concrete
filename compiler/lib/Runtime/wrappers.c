#include "concretelang/Runtime/wrappers.h"
#include <assert.h>
#include <stdio.h>

struct ForeignPlaintextList_u64 *memref_runtime_foreign_plaintext_list_u64(
    uint64_t *allocated, uint64_t *aligned, uint64_t offset, uint64_t size,
    uint64_t stride, uint32_t precision) {

  assert(stride == 1 && "Runtime: stride not equal to 1, check "
                        "runtime_foreign_plaintext_list_u64");

  // Encode table values in u64
  uint64_t *encoded_table = malloc(size * sizeof(uint64_t));
  for (uint64_t i = 0; i < size; i++) {
    encoded_table[i] = (aligned + offset)[i] << (64 - precision - 1);
  }
  return foreign_plaintext_list_u64(encoded_table, size);
  // TODO: is it safe to free after creating plaintext_list?
}

void memref_add_lwe_ciphertexts_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *ct1_allocated, uint64_t *ct1_aligned,
    uint64_t ct1_offset, uint64_t ct1_size, uint64_t ct1_stride) {
  assert(out_size == ct0_size && out_size == ct1_size &&
         "size of lwe buffer are incompatible");
  LweDimension lwe_dimension = {out_size - 1};
  add_two_lwe_ciphertexts_u64(out_aligned + out_offset,
                              ct0_aligned + ct0_offset,
                              ct1_aligned + ct1_offset, lwe_dimension);
}

void memref_add_plaintext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t plaintext) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  LweDimension lwe_dimension = {out_size - 1};
  add_plaintext_to_lwe_ciphertext_u64(out_aligned + out_offset,
                                      ct0_aligned + ct0_offset, plaintext,
                                      lwe_dimension);
}

void memref_mul_cleartext_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t cleartext) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  LweDimension lwe_dimension = {out_size - 1};
  mul_cleartext_lwe_ciphertext_u64(out_aligned + out_offset,
                                   ct0_aligned + ct0_offset, cleartext,
                                   lwe_dimension);
}

void memref_negate_lwe_ciphertext_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride) {
  assert(out_size == ct0_size && "size of lwe buffer are incompatible");
  LweDimension lwe_dimension = {out_size - 1};
  neg_lwe_ciphertext_u64(out_aligned + out_offset, ct0_aligned + ct0_offset,
                         lwe_dimension);
}

void memref_keyswitch_lwe_u64(struct LweKeyswitchKey_u64 *keyswitch_key,
                              uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride) {
  bufferized_keyswitch_lwe_u64(keyswitch_key, out_aligned + out_offset,
                               ct0_aligned + ct0_offset);
}

void memref_bootstrap_lwe_u64(struct LweBootstrapKey_u64 *bootstrap_key,
                              uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride,
                              struct GlweCiphertext_u64 *accumulator) {
  bufferized_bootstrap_lwe_u64(bootstrap_key, out_aligned + out_offset,
                               ct0_aligned + ct0_offset, accumulator);
}
