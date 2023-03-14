#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../include/device.h"
#include "concrete-cpu.h"
#include <functional>

uint64_t *generate_plaintexts(uint64_t payload_modulus, uint64_t delta,
                              int number_of_inputs, const unsigned repetitions,
                              const unsigned samples);

uint64_t *generate_identity_lut_pbs(int polynomial_size, int glwe_dimension,
                                    int message_modulus, int carry_modulus,
                                    std::function<uint64_t(uint64_t)> func);

uint64_t *generate_identity_lut_cmux_tree(int polynomial_size, int num_lut,
                                          int tau, int delta_log);

void generate_lwe_secret_keys(uint64_t **lwe_sk_array, int lwe_dimension,
                              Csprng *csprng, const unsigned repetitions);

void generate_glwe_secret_keys(uint64_t **glwe_sk_array, int glwe_dimension,
                               int polynomial_size, Csprng *csprng,
                               const unsigned repetitions);

void generate_lwe_bootstrap_keys(
    cudaStream_t *stream, int gpu_index, double **d_fourier_bsk_array,
    uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array, int lwe_dimension,
    int glwe_dimension, int polynomial_size, int pbs_level, int pbs_base_log,
    Csprng *csprng, double variance, const unsigned repetitions);

void generate_lwe_keyswitch_keys(cudaStream_t *stream, int gpu_index,
                                 uint64_t **d_ksk_array,
                                 uint64_t *lwe_sk_in_array,
                                 uint64_t *lwe_sk_out_array,
                                 int input_lwe_dimension,
                                 int output_lwe_dimension, int ksk_level,
                                 int ksk_base_log, Csprng *csprng,
                                 double variance, const unsigned repetitions);

void generate_lwe_private_functional_keyswitch_key_lists(
    cudaStream_t *stream, int gpu_index, uint64_t **d_pksk_array,
    uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
    int input_lwe_dimension, int output_glwe_dimension,
    int output_polynomial_size, int pksk_level, int pksk_base_log,
    Csprng *csprng, double variance, const unsigned repetitions);

uint64_t closest_representable(uint64_t input, int level_count, int base_log);

uint64_t *bit_decompose_value(uint64_t value, int r);

#endif