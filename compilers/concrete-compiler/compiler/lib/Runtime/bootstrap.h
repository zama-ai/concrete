#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H

#include "concrete-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void memref_bootstrap_lwe_u64_c(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dimension, uint32_t polynomial_size,
    uint32_t decomposition_level_count, uint32_t decomposition_base_log,
    uint32_t glwe_dimension, uint32_t bsk_index, const c64 *bootstrap_key,
    const Fft *fft);
/* void external_product_c(const struct Fft *fft, const c64 *bootstrap_key_ggsw,
 */
/*                         uint64_t *ct1_closest_representable, uint8_t
 * *scratch, */
/*                         size_t scratch_size, c64 *output_fft_buffer, */
/*                         uint64_t *ct0, uint32_t polynomial_size, */
/*                         uint32_t decomposition_level_count, */
/*                         uint32_t decomposition_base_log, */
/*                         uint32_t glwe_dimension); */

#ifdef __cplusplus
}
#endif

#endif
