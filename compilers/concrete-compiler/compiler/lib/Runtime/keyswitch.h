#ifndef RUNTIME_KEYSWITCH_H
#define RUNTIME_KEYSWITCH_H

#ifdef __cplusplus
extern "C" {
#endif

void keyswitch_lwe_c(uint64_t *out_allocated, uint64_t *out_aligned,
                           uint64_t out_offset, uint64_t out_size,
                           uint64_t out_stride, uint64_t *ct0_allocated,
                           uint64_t *ct0_aligned, uint64_t ct0_offset,
                           uint64_t ct0_size, uint64_t ct0_stride,
                           uint32_t decomposition_level_count,
                           uint32_t decomposition_base_log,
                           uint32_t input_dimension, uint32_t output_dimension,
                           const uint64_t *ksk);

#ifdef __cplusplus
}
#endif

#endif
