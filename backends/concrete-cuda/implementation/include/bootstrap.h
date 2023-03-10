#ifndef CUDA_BOOTSTRAP_H
#define CUDA_BOOTSTRAP_H

#include <cstdint>

extern "C" {

void cuda_convert_lwe_bootstrap_key_32(void *dest, void *src, void *v_stream,
                                       uint32_t gpu_index,
                                       uint32_t input_lwe_dim,
                                       uint32_t glwe_dim, uint32_t level_count,
                                       uint32_t polynomial_size);

void cuda_convert_lwe_bootstrap_key_64(void *dest, void *src, void *v_stream,
                                       uint32_t gpu_index,
                                       uint32_t input_lwe_dim,
                                       uint32_t glwe_dim, uint32_t level_count,
                                       uint32_t polynomial_size);

void scratch_cuda_bootstrap_amortized_32(void *v_stream, uint32_t gpu_index,
                                         int8_t **pbs_buffer,
                                         uint32_t glwe_dimension,
                                         uint32_t polynomial_size,
                                         uint32_t input_lwe_ciphertext_count,
                                         uint32_t max_shared_memory,
                                         bool allocate_gpu_memory);

void scratch_cuda_bootstrap_amortized_64(void *v_stream, uint32_t gpu_index,
                                         int8_t **pbs_buffer,
                                         uint32_t glwe_dimension,
                                         uint32_t polynomial_size,
                                         uint32_t input_lwe_ciphertext_count,
                                         uint32_t max_shared_memory,
                                         bool allocate_gpu_memory);

void cuda_bootstrap_amortized_lwe_ciphertext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples, uint32_t num_lut_vectors, uint32_t lwe_idx,
    uint32_t max_shared_memory);

void cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples, uint32_t num_lut_vectors, uint32_t lwe_idx,
    uint32_t max_shared_memory);

void cleanup_cuda_bootstrap_amortized(void *v_stream, uint32_t gpu_index,
                                      int8_t **pbs_buffer);

void scratch_cuda_bootstrap_low_latency_32(
    void *v_stream, uint32_t gpu_index, int8_t **pbs_buffer,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory,
    bool allocate_gpu_memory);

void scratch_cuda_bootstrap_low_latency_64(
    void *v_stream, uint32_t gpu_index, int8_t **pbs_buffer,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t max_shared_memory,
    bool allocate_gpu_memory);

void cuda_bootstrap_low_latency_lwe_ciphertext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples, uint32_t num_lut_vectors, uint32_t lwe_idx,
    uint32_t max_shared_memory);

void cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lut_vector,
    void *lut_vector_indexes, void *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t num_samples, uint32_t num_lut_vectors, uint32_t lwe_idx,
    uint32_t max_shared_memory);

void cleanup_cuda_bootstrap_low_latency(void *v_stream, uint32_t gpu_index,
                                        int8_t **pbs_buffer);

void scratch_cuda_circuit_bootstrap_vertical_packing_32(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_vp_buffer,
    uint32_t *cbs_delta_log, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t level_count_cbs,
    uint32_t number_of_inputs, uint32_t tau, uint32_t max_shared_memory,
    bool allocate_gpu_memory);

void scratch_cuda_circuit_bootstrap_vertical_packing_64(
    void *v_stream, uint32_t gpu_index, int8_t **cbs_vp_buffer,
    uint32_t *cbs_delta_log, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t level_count_cbs,
    uint32_t number_of_inputs, uint32_t tau, uint32_t max_shared_memory,
    bool allocate_gpu_memory);

void scratch_cuda_wop_pbs_32(
    void *v_stream, uint32_t gpu_index, int8_t **wop_pbs_buffer,
    uint32_t *delta_log, uint32_t *cbs_delta_log, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t level_count_cbs,
    uint32_t level_count_bsk,
    uint32_t number_of_bits_of_message_including_padding,
    uint32_t number_of_bits_to_extract, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory);

void scratch_cuda_wop_pbs_64(
    void *v_stream, uint32_t gpu_index, int8_t **wop_pbs_buffer,
    uint32_t *delta_log, uint32_t *cbs_delta_log, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t level_count_cbs,
    uint32_t level_count_bsk,
    uint32_t number_of_bits_of_message_including_padding,
    uint32_t number_of_bits_to_extract, uint32_t number_of_inputs,
    uint32_t max_shared_memory, bool allocate_gpu_memory);

void cuda_circuit_bootstrap_vertical_packing_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *fourier_bsk, void *cbs_fpksk, void *lut_vector, int8_t *cbs_vp_buffer,
    uint32_t cbs_delta_log, uint32_t polynomial_size, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t level_count_bsk, uint32_t base_log_bsk,
    uint32_t level_count_pksk, uint32_t base_log_pksk, uint32_t level_count_cbs,
    uint32_t base_log_cbs, uint32_t number_of_inputs, uint32_t lut_number,
    uint32_t max_shared_memory);

void cuda_wop_pbs_64(void *v_stream, uint32_t gpu_index, void *lwe_array_out,
                     void *lwe_array_in, void *lut_vector, void *fourier_bsk,
                     void *ksk, void *cbs_fpksk, int8_t *wop_pbs_buffer,
                     uint32_t cbs_delta_log, uint32_t glwe_dimension,
                     uint32_t lwe_dimension, uint32_t polynomial_size,
                     uint32_t base_log_bsk, uint32_t level_count_bsk,
                     uint32_t base_log_ksk, uint32_t level_count_ksk,
                     uint32_t base_log_pksk, uint32_t level_count_pksk,
                     uint32_t base_log_cbs, uint32_t level_count_cbs,
                     uint32_t number_of_bits_of_message_including_padding,
                     uint32_t number_of_bits_to_extract, uint32_t delta_log,
                     uint32_t number_of_inputs, uint32_t max_shared_memory);

void cleanup_cuda_wop_pbs(void *v_stream, uint32_t gpu_index,
                          int8_t **wop_pbs_buffer);

void cleanup_cuda_circuit_bootstrap_vertical_packing(void *v_stream,
                                                     uint32_t gpu_index,
                                                     int8_t **cbs_vp_buffer);
}

#ifdef __CUDACC__
__device__ inline int get_start_ith_ggsw(int i, uint32_t polynomial_size,
                                         int glwe_dimension,
                                         uint32_t level_count);

template <typename T>
__device__ T *get_ith_mask_kth_block(T *ptr, int i, int k, int level,
                                     uint32_t polynomial_size,
                                     int glwe_dimension, uint32_t level_count);

template <typename T>
__device__ T *get_ith_body_kth_block(T *ptr, int i, int k, int level,
                                     uint32_t polynomial_size,
                                     int glwe_dimension, uint32_t level_count);
#endif

#endif // CUDA_BOOTSTRAP_H
