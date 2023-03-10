#ifndef CUDA_CIRCUIT_BOOTSTRAP_H
#define CUDA_CIRCUIT_BOOTSTRAP_H

#include <cstdint>

extern "C" {

void scratch_cuda_circuit_bootstrap_32(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory);

void scratch_cuda_circuit_bootstrap_64(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_buffer,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t level_count_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory);

void cuda_circuit_bootstrap_32(
    void *v_stream, uint32_t gpu_index, void *ggsw_out, void *lwe_array_in,
    void *fourier_bsk, void *fp_ksk_array, void *lut_vector_indexes,
    int8_t *cbs_buffer, uint32_t delta_log, uint32_t polynomial_size,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t level_bsk,
    uint32_t base_log_bsk, uint32_t level_pksk, uint32_t base_log_pksk,
    uint32_t level_cbs, uint32_t base_log_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory);

void cuda_circuit_bootstrap_64(
    void *v_stream, uint32_t gpu_index, void *ggsw_out, void *lwe_array_in,
    void *fourier_bsk, void *fp_ksk_array, void *lut_vector_indexes,
    int8_t *cbs_buffer, uint32_t delta_log, uint32_t polynomial_size,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t level_bsk,
    uint32_t base_log_bsk, uint32_t level_pksk, uint32_t base_log_pksk,
    uint32_t level_cbs, uint32_t base_log_cbs, uint32_t number_of_inputs,
    uint32_t max_shared_memory);

void cleanup_cuda_circuit_bootstrap(void *v_stream, uint32_t gpu_index,
                                    int8_t **cbs_buffer);
}

#endif // CUDA_CIRCUIT_BOOTSTRAP_H
