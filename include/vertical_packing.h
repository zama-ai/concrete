#ifndef VERTICAL_PACKING_H
#define VERTICAL_PACKING_H

#include <cstdint>

extern "C" {

void scratch_cuda_cmux_tree_32(void *v_stream, uint32_t gpu_index,
                               int8_t **cmux_tree_buffer,
                               uint32_t glwe_dimension,
                               uint32_t polynomial_size, uint32_t level_count,
                               uint32_t r, uint32_t tau,
                               uint32_t max_shared_memory,
                               bool allocate_gpu_memory);

void scratch_cuda_cmux_tree_64(void *v_stream, uint32_t gpu_index,
                               int8_t **cmux_tree_buffer,
                               uint32_t glwe_dimension,
                               uint32_t polynomial_size, uint32_t level_count,
                               uint32_t r, uint32_t tau,
                               uint32_t max_shared_memory,
                               bool allocate_gpu_memory);

void cuda_cmux_tree_32(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector,
                       int8_t *cmux_tree_buffer, uint32_t glwe_dimension,
                       uint32_t polynomial_size, uint32_t base_log,
                       uint32_t level_count, uint32_t r, uint32_t tau,
                       uint32_t max_shared_memory);

void cuda_cmux_tree_64(void *v_stream, uint32_t gpu_index, void *glwe_array_out,
                       void *ggsw_in, void *lut_vector,
                       int8_t *cmux_tree_buffer, uint32_t glwe_dimension,
                       uint32_t polynomial_size, uint32_t base_log,
                       uint32_t level_count, uint32_t r, uint32_t tau,
                       uint32_t max_shared_memory);

void cleanup_cuda_cmux_tree(void *v_stream, uint32_t gpu_index,
                            int8_t **cmux_tree_buffer);

void cuda_blind_rotate_and_sample_extraction_64(
    void *v_stream, uint32_t gpu_index, void *lwe_out, void *ggsw_in,
    void *lut_vector, uint32_t mbr_size, uint32_t tau, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t l_gadget,
    uint32_t max_shared_memory);
}

#endif // VERTICAL_PACKING_H
