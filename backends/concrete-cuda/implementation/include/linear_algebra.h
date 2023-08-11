#ifndef CUDA_LINALG_H_
#define CUDA_LINALG_H_

#include <cstdint>

extern "C" {

// Three types of pbs are available for integer multiplication
enum PBS_TYPE { MULTI_BIT = 0, LOW_LAT = 1, AMORTIZED = 2 };

void cuda_negate_lwe_ciphertext_vector_32(void *v_stream, uint32_t gpu_index,
                                          void *lwe_array_out,
                                          void *lwe_array_in,
                                          uint32_t input_lwe_dimension,
                                          uint32_t input_lwe_ciphertext_count);
void cuda_negate_lwe_ciphertext_vector_64(void *v_stream, uint32_t gpu_index,
                                          void *lwe_array_out,
                                          void *lwe_array_in,
                                          uint32_t input_lwe_dimension,
                                          uint32_t input_lwe_ciphertext_count);
void cuda_add_lwe_ciphertext_vector_32(void *v_stream, uint32_t gpu_index,
                                       void *lwe_array_out,
                                       void *lwe_array_in_1,
                                       void *lwe_array_in_2,
                                       uint32_t input_lwe_dimension,
                                       uint32_t input_lwe_ciphertext_count);
void cuda_add_lwe_ciphertext_vector_64(void *v_stream, uint32_t gpu_index,
                                       void *lwe_array_out,
                                       void *lwe_array_in_1,
                                       void *lwe_array_in_2,
                                       uint32_t input_lwe_dimension,
                                       uint32_t input_lwe_ciphertext_count);
void cuda_add_lwe_ciphertext_vector_plaintext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *plaintext_array_in, uint32_t input_lwe_dimension,
    uint32_t input_lwe_ciphertext_count);
void cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *plaintext_array_in, uint32_t input_lwe_dimension,
    uint32_t input_lwe_ciphertext_count);
void cuda_mult_lwe_ciphertext_vector_cleartext_vector_32(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *cleartext_array_in, uint32_t input_lwe_dimension,
    uint32_t input_lwe_ciphertext_count);
void cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
    void *v_stream, uint32_t gpu_index, void *lwe_array_out, void *lwe_array_in,
    void *cleartext_array_in, uint32_t input_lwe_dimension,
    uint32_t input_lwe_ciphertext_count);

void scratch_cuda_integer_mult_radix_ciphertext_kb_64(
    void *v_stream, uint32_t gpu_index, void *mem_ptr, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level,
    uint32_t ks_base_log, uint32_t ks_level, uint32_t num_blocks,
    PBS_TYPE pbs_type, uint32_t max_shared_memory, bool allocate_gpu_memory);

void cuda_integer_mult_radix_ciphertext_kb_64(
    void *v_stream, uint32_t gpu_index, void *radix_lwe_out,
    void *radix_lwe_left, void *radix_lwe_right, uint32_t *ct_degree_out,
    uint32_t *ct_degree_left, uint32_t *ct_degree_right, void *bsk, void *ksk,
    void *mem_ptr, uint32_t message_modulus, uint32_t carry_modulus,
    uint32_t glwe_dimension, uint32_t lwe_dimension, uint32_t polynomial_size,
    uint32_t pbs_base_log, uint32_t pbs_level, uint32_t ks_base_log,
    uint32_t ks_level, uint32_t num_blocks, PBS_TYPE pbs_type,
    uint32_t max_shared_memory);

void scratch_cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
    void *mem_ptr, void *bsk, void *ksk, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level,
    uint32_t ks_base_log, uint32_t ks_level, uint32_t num_blocks,
    PBS_TYPE pbs_type, uint32_t max_shared_memory, bool allocate_gpu_memory);

void cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
    void *radix_lwe_out, void *radix_lwe_left, void *radix_lwe_right,
    uint32_t *ct_degree_out, uint32_t *ct_degree_left,
    uint32_t *ct_degree_right, void *bsk, void *ksk, void *mem_ptr,
    uint32_t message_modulus, uint32_t carry_modulus, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t pbs_base_log,
    uint32_t pbs_level, uint32_t ks_base_log, uint32_t ks_level,
    uint32_t num_blocks, PBS_TYPE pbs_type, uint32_t max_shared_memory);


}




#endif // CUDA_LINALG_H_
