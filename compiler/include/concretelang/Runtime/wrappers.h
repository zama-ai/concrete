// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_WRAPPERS_H
#define CONCRETELANG_RUNTIME_WRAPPERS_H

#include "concrete-ffi.h"

struct ForeignPlaintextList_u64 *memref_runtime_foreign_plaintext_list_u64(
    uint64_t *allocated, uint64_t *aligned, uint64_t offset, uint64_t size,
    uint64_t stride, uint32_t precision);

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

void memref_keyswitch_lwe_u64(struct LweKeyswitchKey_u64 *keyswitch_key,
                              uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride);

void memref_bootstrap_lwe_u64(struct LweBootstrapKey_u64 *bootstrap_key,
                              uint64_t *out_allocated, uint64_t *out_aligned,
                              uint64_t out_offset, uint64_t out_size,
                              uint64_t out_stride, uint64_t *ct0_allocated,
                              uint64_t *ct0_aligned, uint64_t ct0_offset,
                              uint64_t ct0_size, uint64_t ct0_stride,
                              struct GlweCiphertext_u64 *accumulator);

#endif
