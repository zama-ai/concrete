// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

/// Define the API exposed to the compiler for streaming code generation.

#ifndef CONCRETELANG_STREAM_EMULATOR_API_H
#define CONCRETELANG_STREAM_EMULATOR_API_H
#include <cstddef>
#include <cstdint>
#include <cstdlib>

typedef enum stream_type {
  TS_STREAM_TYPE_X86_TO_TOPO_LSAP,
  TS_STREAM_TYPE_TOPO_TO_TOPO_LSAP,
  TS_STREAM_TYPE_TOPO_TO_X86_LSAP,
  TS_STREAM_TYPE_TOPO_TO_BOTH,
  TS_STREAM_TYPE_X86_TO_X86_LSAP
} stream_type;
template <size_t N> struct MemRefDescriptor {
  uint64_t *allocated;
  uint64_t *aligned;
  size_t offset;
  size_t sizes[N];
  size_t strides[N];
};

extern "C" {
void *stream_emulator_init();
void stream_emulator_run(void *dfg);
void stream_emulator_delete(void *dfg);

void stream_emulator_make_memref_add_lwe_ciphertexts_u64_process(void *dfg,
                                                                 void *sin1,
                                                                 void *sin2,
                                                                 void *sout);
void stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_negate_lwe_ciphertext_u64_process(void *dfg,
                                                                   void *sin1,
                                                                   void *sout);
void stream_emulator_make_memref_keyswitch_lwe_u64_process(
    void *dfg, void *sin1, void *sout, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t output_size,
    uint32_t ksk_index, void *context);
void stream_emulator_make_memref_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context);

void stream_emulator_make_memref_batched_add_lwe_ciphertexts_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_batched_add_plaintext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_batched_add_plaintext_cst_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_batched_mul_cleartext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_batched_mul_cleartext_cst_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout);
void stream_emulator_make_memref_batched_negate_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sout);
void stream_emulator_make_memref_batched_keyswitch_lwe_u64_process(
    void *dfg, void *sin1, void *sout, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t output_size,
    uint32_t ksk_index, void *context);
void stream_emulator_make_memref_batched_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context);
void stream_emulator_make_memref_batched_mapped_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context);

void *stream_emulator_make_uint64_stream(const char *name, stream_type stype);
void stream_emulator_put_uint64(void *stream, uint64_t e);
uint64_t stream_emulator_get_uint64(void *stream);

void *stream_emulator_make_memref_stream(const char *name, stream_type stype);
void stream_emulator_put_memref(void *stream, uint64_t *allocated,
                                uint64_t *aligned, uint64_t offset,
                                uint64_t size, uint64_t stride,
                                uint64_t data_ownership);
void stream_emulator_get_memref(void *stream, uint64_t *out_allocated,
                                uint64_t *out_aligned, uint64_t out_offset,
                                uint64_t out_size, uint64_t out_stride);

void *stream_emulator_make_memref_batch_stream(const char *name,
                                               stream_type stype);
void stream_emulator_put_memref_batch(void *stream, uint64_t *allocated,
                                      uint64_t *aligned, uint64_t offset,
                                      uint64_t size0, uint64_t size1,
                                      uint64_t stride0, uint64_t stride1,
                                      uint64_t data_ownership);
void stream_emulator_get_memref_batch(void *stream, uint64_t *out_allocated,
                                      uint64_t *out_aligned,
                                      uint64_t out_offset, uint64_t out_size0,
                                      uint64_t out_size1, uint64_t out_stride0,
                                      uint64_t out_stride1);
}

#endif
