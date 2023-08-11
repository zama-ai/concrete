#ifndef SETUP_AND_TEARDOWN_H
#define SETUP_AND_TEARDOWN_H

#include <bit_extraction.h>
#include <bootstrap.h>
#include <bootstrap_multibit.h>
#include <circuit_bootstrap.h>
#include <concrete-cpu.h>
#include <device.h>
#include <keyswitch.h>
#include <linear_algebra.h>
#include <utils.h>
#include <vertical_packing.h>

template <typename Torus> struct int_mul_memory {
  Torus *vector_result_sb;
  Torus *block_mul_res;
  Torus *small_lwe_vector;
  Torus *lwe_pbs_out_array;
  Torus *test_vector_array;
  Torus *message_acc;
  Torus *carry_acc;
  Torus *test_vector_indexes;
  Torus *tvi_message;
  Torus *tvi_carry;
  int8_t *pbs_buffer;

  int p2p_gpu_count = 0;

  cudaStream_t *streams[32];

  int8_t *pbs_buffer_multi_gpu[32];
  Torus *pbs_input_multi_gpu[32];
  Torus *pbs_output_multi_gpu[32];
  Torus *test_vector_multi_gpu[32];
  Torus *tvi_lsb_multi_gpu[32];
  Torus *tvi_msb_multi_gpu[32];
  Torus *tvi_message_multi_gpu[32];
  Torus *tvi_carry_multi_gpu[32];
  Torus *bsk_multi_gpu[32];
  Torus *ksk_multi_gpu[32];

  Torus *device_to_device_buffer[8];

  bool IsAppBuiltAs64() { return sizeof(void *) == 8; }
};


void bootstrap_classical_setup(
    cudaStream_t *stream, Csprng **csprng, uint64_t **lwe_sk_in_array,
    uint64_t **lwe_sk_out_array, double **d_fourier_bsk_array,
    uint64_t **plaintexts, uint64_t **d_lut_pbs_identity,
    uint64_t **d_lut_pbs_indexes, uint64_t **d_lwe_ct_in_array,
    uint64_t **d_lwe_ct_out_array, int lwe_dimension, int glwe_dimension,
    int polynomial_size, double lwe_modular_variance,
    double glwe_modular_variance, int pbs_base_log, int pbs_level,
    int message_modulus, int carry_modulus, int *payload_modulus,
    uint64_t *delta, int number_of_inputs, int repetitions, int samples,
    int gpu_index);
void bootstrap_classical_teardown(
    cudaStream_t *stream, Csprng *csprng, uint64_t *lwe_sk_in_array,
    uint64_t *lwe_sk_out_array, double *d_fourier_bsk_array,
    uint64_t *plaintexts, uint64_t *d_lut_pbs_identity,
    uint64_t *d_lut_pbs_indexes, uint64_t *d_lwe_ct_in_array,
    uint64_t *d_lwe_ct_out_array, int gpu_index);

void bootstrap_multibit_setup(
    cudaStream_t *stream, Csprng **csprng, uint64_t **lwe_sk_in_array,
    uint64_t **lwe_sk_out_array, uint64_t **d_bsk_array, uint64_t **plaintexts,
    uint64_t **d_lut_pbs_identity, uint64_t **d_lut_pbs_indexes,
    uint64_t **d_lwe_ct_in_array, uint64_t **d_lwe_ct_out_array,
    int8_t **pbs_buffer, int lwe_dimension, int glwe_dimension,
    int polynomial_size, int grouping_factor, double lwe_modular_variance,
    double glwe_modular_variance, int pbs_base_log, int pbs_level,
    int message_modulus, int carry_modulus, int *payload_modulus,
    uint64_t *delta, int number_of_inputs, int repetitions, int samples,
    int gpu_index, int chunk_size = 0);

void bootstrap_multibit_teardown(
    cudaStream_t *stream, Csprng *csprng, uint64_t *lwe_sk_in_array,
    uint64_t *lwe_sk_out_array, uint64_t *d_bsk_array, uint64_t *plaintexts,
    uint64_t *d_lut_pbs_identity, uint64_t *d_lut_pbs_indexes,
    uint64_t *d_lwe_ct_in_array, uint64_t *d_lwe_ct_out_array,
    int8_t **pbs_buffer, int gpu_index);

void keyswitch_setup(cudaStream_t *stream, Csprng **csprng,
                     uint64_t **lwe_sk_in_array, uint64_t **lwe_sk_out_array,
                     uint64_t **d_ksk_array, uint64_t **plaintexts,
                     uint64_t **d_lwe_ct_in_array,
                     uint64_t **d_lwe_ct_out_array, int input_lwe_dimension,
                     int output_lwe_dimension, double lwe_modular_variance,
                     int ksk_base_log, int ksk_level, int message_modulus,
                     int carry_modulus, int *payload_modulus, uint64_t *delta,
                     int number_of_inputs, int repetitions, int samples,
                     int gpu_index);
void keyswitch_teardown(cudaStream_t *stream, Csprng *csprng,
                        uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
                        uint64_t *d_ksk_array, uint64_t *plaintexts,
                        uint64_t *d_lwe_ct_in_array,
                        uint64_t *d_lwe_ct_out_array, int gpu_index);

void bit_extraction_setup(
    cudaStream_t **stream, Csprng **csprng, uint64_t **lwe_sk_in_array,
    uint64_t **lwe_sk_out_array, double **d_fourier_bsk_array,
    uint64_t **d_ksk_array, uint64_t **plaintexts, uint64_t **d_lwe_ct_in_array,
    uint64_t **d_lwe_ct_out_array, int8_t **bit_extract_buffer_array,
    int lwe_dimension, int glwe_dimension, int polynomial_size,
    double lwe_modular_variance, double glwe_modular_variance, int ks_base_log,
    int ks_level, int pbs_base_log, int pbs_level,
    uint32_t *number_of_bits_of_message_including_padding_array,
    uint32_t *number_of_bits_to_extract_array, uint32_t *delta_log_array,
    uint64_t *delta_array, int crt_decomposition_size, int repetitions,
    int samples, int gpu_index);

void bit_extraction_teardown(cudaStream_t **stream, Csprng *csprng,
                             uint64_t *lwe_sk_in_array,
                             uint64_t *lwe_sk_out_array,
                             double *d_fourier_bsk_array, uint64_t *d_ksk_array,
                             uint64_t *plaintexts, uint64_t *d_lwe_ct_in_array,
                             uint64_t *d_lwe_ct_out_array,
                             int8_t **bit_extract_buffer_array, int samples,
                             int gpu_index);

void circuit_bootstrap_setup(
    cudaStream_t *stream, Csprng **csprng, uint64_t **lwe_sk_in_array,
    uint64_t **lwe_sk_out_array, double **d_fourier_bsk_array,
    uint64_t **d_pksk_array, uint64_t **plaintexts,
    uint64_t **d_lwe_ct_in_array, uint64_t **d_ggsw_ct_out_array,
    uint64_t **d_lut_vector_indexes, int8_t **cbs_buffer, int lwe_dimension,
    int glwe_dimension, int polynomial_size, double lwe_modular_variance,
    double glwe_modular_variance, int pksk_base_log, int pksk_level,
    int pbs_base_log, int pbs_level, int cbs_level,
    int number_of_bits_of_message_including_padding, int ggsw_size,
    int *delta_log, uint64_t *delta, int number_of_inputs, int repetitions,
    int samples, int gpu_index);

void circuit_bootstrap_teardown(
    cudaStream_t *stream, Csprng *csprng, uint64_t *lwe_sk_in_array,
    uint64_t *lwe_sk_out_array, double *d_fourier_bsk_array,
    uint64_t *d_pksk_array, uint64_t *plaintexts, uint64_t *d_lwe_ct_in_array,
    uint64_t *d_lut_vector_indexes, uint64_t *d_ggsw_ct_out_array,
    int8_t *cbs_buffer, int gpu_index);

void cmux_tree_setup(cudaStream_t *stream, Csprng **csprng, uint64_t **glwe_sk,
                     uint64_t **d_lut_identity, uint64_t **plaintexts,
                     uint64_t **d_ggsw_bit_array, int8_t **cmux_tree_buffer,
                     uint64_t **d_glwe_out, int glwe_dimension,
                     int polynomial_size, int base_log, int level_count,
                     double glwe_modular_variance, int r_lut, int tau,
                     uint32_t *delta_log, int repetitions, int samples,
                     int gpu_index);
void cmux_tree_teardown(cudaStream_t *stream, Csprng **csprng,
                        uint64_t **glwe_sk, uint64_t **d_lut_identity,
                        uint64_t **plaintexts, uint64_t **d_ggsw_bit_array,
                        int8_t **cmux_tree_buffer, uint64_t **d_glwe_out,
                        int gpu_index);

void wop_pbs_setup(cudaStream_t *stream, Csprng **csprng,
                   uint64_t **lwe_sk_in_array, uint64_t **lwe_sk_out_array,
                   uint64_t **d_ksk_array, double **d_fourier_bsk_array,
                   uint64_t **d_pksk_array, uint64_t **plaintexts,
                   uint64_t **d_lwe_ct_in_array, uint64_t **d_lwe_ct_out_array,
                   uint64_t **d_lut_vector, int8_t **wop_pbs_buffer,
                   int lwe_dimension, int glwe_dimension, int polynomial_size,
                   double lwe_modular_variance, double glwe_modular_variance,
                   int ks_base_log, int ks_level, int pksk_base_log,
                   int pksk_level, int pbs_base_log, int pbs_level,
                   int cbs_level, uint32_t *p_array, uint32_t *delta_log_array,
                   int *cbs_delta_log, uint64_t *delta_array, int tau,
                   int repetitions, int samples, int gpu_index);

void wop_pbs_teardown(cudaStream_t *stream, Csprng *csprng,
                      uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
                      uint64_t *d_ksk_array, double *d_fourier_bsk_array,
                      uint64_t *d_pksk_array, uint64_t *plaintexts,
                      uint64_t *d_lwe_ct_in_array, uint64_t *d_lut_vector,
                      uint64_t *d_lwe_ct_out_array, int8_t *wop_pbs_buffer,
                      int gpu_index);

void linear_algebra_setup(cudaStream_t *stream, Csprng **csprng,
                          uint64_t **lwe_sk_array, uint64_t **d_lwe_in_1_ct,
                          uint64_t **d_lwe_in_2_ct, uint64_t **d_lwe_out_ct,
                          uint64_t **lwe_in_1_ct, uint64_t **lwe_in_2_ct,
                          uint64_t **lwe_out_ct, uint64_t **plaintexts_1,
                          uint64_t **plaintexts_2, uint64_t **d_plaintext_2,
                          uint64_t **d_plaintexts_2_mul, int lwe_dimension,
                          double noise_variance, int payload_modulus,
                          uint64_t delta, int number_of_inputs, int repetitions,
                          int samples, int gpu_index);

void linear_algebra_teardown(cudaStream_t *stream, Csprng **csprng,
                             uint64_t **lwe_sk_array, uint64_t **d_lwe_in_1_ct,
                             uint64_t **d_lwe_in_2_ct, uint64_t **d_lwe_out_ct,
                             uint64_t **lwe_in_1_ct, uint64_t **lwe_in_2_ct,
                             uint64_t **lwe_out_ct, uint64_t **plaintexts_1,
                             uint64_t **plaintexts_2, uint64_t **d_plaintext_2,
                             uint64_t **d_plaintexts_2_mul, int gpu_index);
void fft_setup(cudaStream_t *stream, double **poly1, double **poly2,
               double2 **h_cpoly1, double2 **h_cpoly2, double2 **d_cpoly1,
               double2 **d_cpoly2, size_t polynomial_size, int samples,
               int gpu_index);

void fft_teardown(cudaStream_t *stream, double *poly1, double *poly2,
                  double2 *h_cpoly1, double2 *h_cpoly2, double2 *d_cpoly1,
                  double2 *d_cpoly2, int gpu_index);


void integer_multiplication_setup(
    cudaStream_t *stream, Csprng **csprng, uint64_t **lwe_sk_in_array,
    uint64_t **lwe_sk_out_array, void **d_bsk_array, uint64_t **d_ksk_array,
    uint64_t **plaintexts_1, uint64_t **plaintexts_2,
    uint64_t **d_lwe_ct_in_array_1, uint64_t **d_lwe_ct_in_array_2,
    uint64_t **d_lwe_ct_out_array, int_mul_memory<uint64_t> *mem_ptr,
    int lwe_dimension, int glwe_dimension, int polynomial_size,
    double lwe_modular_variance, double glwe_modular_variance, int pbs_base_log,
    int pbs_level, int ksk_base_log, int ksk_level, int total_message_bits,
    int number_of_blocks, int message_modulus, int carry_modulus,
    uint64_t *delta, int repetitions, int samples, PBS_TYPE pbs_type, int gpu_index);

void integer_multiplication_teardown(
    cudaStream_t *stream, Csprng *csprng, uint64_t *lwe_sk_in_array,
    uint64_t *lwe_sk_out_array, void *d_bsk_array, uint64_t *d_ksk_array,
    uint64_t *plaintexts_1, uint64_t *plaintexts_2,
    uint64_t *d_lwe_ct_in_array_1, uint64_t *d_lwe_ct_in_array_2,
    uint64_t *d_lwe_ct_out_array, int_mul_memory<uint64_t> *mem_ptr);


#endif // SETUP_AND_TEARDOWN_H
