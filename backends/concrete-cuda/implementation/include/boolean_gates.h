#ifndef CUDA_BOOLEAN_GATES_H
#define CUDA_BOOLEAN_GATES_H

#include <cstdint>

extern "C" {

void cuda_boolean_not_32(void *v_stream, uint32_t gpu_index,
                         void *lwe_array_out, void *lwe_array_in,
                         uint32_t input_lwe_dimension,
                         uint32_t input_lwe_ciphertext_count);

void cuda_boolean_and_32(void *v_stream, uint32_t gpu_index,
                         void *lwe_array_out, void *lwe_array_in_1,
                         void *lwe_array_in_2, void *bootstrapping_key,
                         void *ksk, uint32_t input_lwe_dimension,
                         uint32_t glwe_dimension, uint32_t polynomial_size,
                         uint32_t pbs_base_log, uint32_t pbs_level_count,
                         uint32_t ks_base_log, uint32_t ks_level_count,
                         uint32_t input_lwe_ciphertext_count,
                         uint32_t max_shared_memory);

void cuda_boolean_nand_32(void *v_stream, uint32_t gpu_index,
                          void *lwe_array_out, void *lwe_array_in_1,
                          void *lwe_array_in_2, void *bootstrapping_key,
                          void *ksk, uint32_t input_lwe_dimension,
                          uint32_t glwe_dimension, uint32_t polynomial_size,
                          uint32_t pbs_base_log, uint32_t pbs_level_count,
                          uint32_t ks_base_log, uint32_t ks_level_count,
                          uint32_t input_lwe_ciphertext_count,
                          uint32_t max_shared_memory);

void cuda_boolean_nor_32(void *v_stream, uint32_t gpu_index,
                         void *lwe_array_out, void *lwe_array_in_1,
                         void *lwe_array_in_2, void *bootstrapping_key,
                         void *ksk, uint32_t input_lwe_dimension,
                         uint32_t glwe_dimension, uint32_t polynomial_size,
                         uint32_t pbs_base_log, uint32_t pbs_level_count,
                         uint32_t ks_base_log, uint32_t ks_level_count,
                         uint32_t input_lwe_ciphertext_count,
                         uint32_t max_shared_memory);

void cuda_boolean_or_32(void *v_stream, uint32_t gpu_index, void *lwe_array_out,
                        void *lwe_array_in_1, void *lwe_array_in_2,
                        void *bootstrapping_key, void *ksk,
                        uint32_t input_lwe_dimension, uint32_t glwe_dimension,
                        uint32_t polynomial_size, uint32_t pbs_base_log,
                        uint32_t pbs_level_count, uint32_t ks_base_log,
                        uint32_t ks_level_count,
                        uint32_t input_lwe_ciphertext_count,
                        uint32_t max_shared_memory);

void cuda_boolean_xor_32(void *v_stream, uint32_t gpu_index,
                         void *lwe_array_out, void *lwe_array_in_1,
                         void *lwe_array_in_2, void *bootstrapping_key,
                         void *ksk, uint32_t input_lwe_dimension,
                         uint32_t glwe_dimension, uint32_t polynomial_size,
                         uint32_t pbs_base_log, uint32_t pbs_level_count,
                         uint32_t ks_base_log, uint32_t ks_level_count,
                         uint32_t input_lwe_ciphertext_count,
                         uint32_t max_shared_memory);

void cuda_boolean_xnor_32(void *v_stream, uint32_t gpu_index,
                          void *lwe_array_out, void *lwe_array_in_1,
                          void *lwe_array_in_2, void *bootstrapping_key,
                          void *ksk, uint32_t input_lwe_dimension,
                          uint32_t glwe_dimension, uint32_t polynomial_size,
                          uint32_t pbs_base_log, uint32_t pbs_level_count,
                          uint32_t ks_base_log, uint32_t ks_level_count,
                          uint32_t input_lwe_ciphertext_count,
                          uint32_t max_shared_memory);
}

#endif // CUDA_BOOLAN_GATES_H
