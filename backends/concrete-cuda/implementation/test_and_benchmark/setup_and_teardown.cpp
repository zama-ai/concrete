#include <cmath>
#include <random>
#include <setup_and_teardown.h>

void bootstrap_setup(cudaStream_t *stream, Csprng **csprng,
                     uint64_t **lwe_sk_in_array, uint64_t **lwe_sk_out_array,
                     double **d_fourier_bsk_array, uint64_t **plaintexts,
                     uint64_t **d_lut_pbs_identity,
                     uint64_t **d_lut_pbs_indexes, uint64_t **d_lwe_ct_in_array,
                     uint64_t **d_lwe_ct_out_array, int lwe_dimension,
                     int glwe_dimension, int polynomial_size,
                     double lwe_modular_variance, double glwe_modular_variance,
                     int pbs_base_log, int pbs_level, int message_modulus,
                     int carry_modulus, int *payload_modulus, uint64_t *delta,
                     int number_of_inputs, int repetitions, int samples,
                     int gpu_index) {

  *payload_modulus = message_modulus * carry_modulus;
  // Value of the shift we multiply our messages by
  *delta = ((uint64_t)(1) << 63) / (uint64_t)(*payload_modulus);

  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  // Generate the keys
  generate_lwe_secret_keys(lwe_sk_in_array, lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_secret_keys(lwe_sk_out_array, glwe_dimension * polynomial_size,
                           *csprng, repetitions);
  generate_lwe_bootstrap_keys(
      stream, gpu_index, d_fourier_bsk_array, *lwe_sk_in_array,
      *lwe_sk_out_array, lwe_dimension, glwe_dimension, polynomial_size,
      pbs_level, pbs_base_log, *csprng, glwe_modular_variance, repetitions);
  *plaintexts = generate_plaintexts(*payload_modulus, *delta, number_of_inputs,
                                    repetitions, samples);

  // Create the LUT
  uint64_t *lut_pbs_identity = generate_identity_lut_pbs(
      polynomial_size, glwe_dimension, message_modulus, carry_modulus,
      [](int x) -> int { return x; });
  uint64_t *lwe_ct_in_array =
      (uint64_t *)malloc((lwe_dimension + 1) * number_of_inputs * repetitions *
                         samples * sizeof(uint64_t));
  // Create the input/output ciphertexts
  for (int r = 0; r < repetitions; r++) {
    uint64_t *lwe_sk_in = *lwe_sk_in_array + (ptrdiff_t)(r * lwe_dimension);
    for (int s = 0; s < samples; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = (*plaintexts)[r * samples * number_of_inputs +
                                           s * number_of_inputs + i];
        uint64_t *lwe_ct_in =
            lwe_ct_in_array + (ptrdiff_t)((r * samples * number_of_inputs +
                                           s * number_of_inputs + i) *
                                          (lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk_in, lwe_ct_in, plaintext, lwe_dimension,
            lwe_modular_variance, *csprng, &CONCRETE_CSPRNG_VTABLE);
      }
    }
  }

  // Initialize and copy things in/to the device
  *d_lut_pbs_identity = (uint64_t *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(uint64_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(*d_lut_pbs_identity, lut_pbs_identity,
                           polynomial_size * (glwe_dimension + 1) *
                               sizeof(uint64_t),
                           stream, gpu_index);
  *d_lut_pbs_indexes = (uint64_t *)cuda_malloc_async(
      number_of_inputs * sizeof(uint64_t), stream, gpu_index);
  cuda_memset_async(*d_lut_pbs_indexes, 0, number_of_inputs * sizeof(uint64_t),
                    stream, gpu_index);

  // Input and output LWEs
  *d_lwe_ct_out_array =
      (uint64_t *)cuda_malloc_async((glwe_dimension * polynomial_size + 1) *
                                        number_of_inputs * sizeof(uint64_t),
                                    stream, gpu_index);
  *d_lwe_ct_in_array = (uint64_t *)cuda_malloc_async(
      (lwe_dimension + 1) * number_of_inputs * repetitions * samples *
          sizeof(uint64_t),
      stream, gpu_index);

  cuda_memcpy_async_to_gpu(*d_lwe_ct_in_array, lwe_ct_in_array,
                           repetitions * samples * number_of_inputs *
                               (lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);

  cuda_synchronize_stream(stream);

  free(lwe_ct_in_array);
  free(lut_pbs_identity);
}

void bootstrap_teardown(cudaStream_t *stream, Csprng *csprng,
                        uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
                        double *d_fourier_bsk_array, uint64_t *plaintexts,
                        uint64_t *d_lut_pbs_identity,
                        uint64_t *d_lut_pbs_indexes,
                        uint64_t *d_lwe_ct_in_array,
                        uint64_t *d_lwe_ct_out_array, int gpu_index) {
  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(csprng);
  free(csprng);
  free(lwe_sk_in_array);
  free(lwe_sk_out_array);
  free(plaintexts);

  cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
  cuda_drop_async(d_lut_pbs_identity, stream, gpu_index);
  cuda_drop_async(d_lut_pbs_indexes, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_out_array, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

void keyswitch_setup(cudaStream_t *stream, Csprng **csprng,
                     uint64_t **lwe_sk_in_array, uint64_t **lwe_sk_out_array,
                     uint64_t **d_ksk_array, uint64_t **plaintexts,
                     uint64_t **d_lwe_ct_in_array,
                     uint64_t **d_lwe_ct_out_array, int input_lwe_dimension,
                     int output_lwe_dimension, double lwe_modular_variance,
                     int ksk_base_log, int ksk_level, int message_modulus,
                     int carry_modulus, int *payload_modulus, uint64_t *delta,
                     int number_of_inputs, int repetitions, int samples,
                     int gpu_index) {

  *payload_modulus = message_modulus * carry_modulus;
  // Value of the shift we multiply our messages by
  *delta = ((uint64_t)(1) << 63) / (uint64_t)(*payload_modulus);

  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  // Generate the keys
  generate_lwe_secret_keys(lwe_sk_in_array, input_lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_secret_keys(lwe_sk_out_array, output_lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_keyswitch_keys(stream, gpu_index, d_ksk_array, *lwe_sk_in_array,
                              *lwe_sk_out_array, input_lwe_dimension,
                              output_lwe_dimension, ksk_level, ksk_base_log,
                              *csprng, lwe_modular_variance, repetitions);
  *plaintexts = generate_plaintexts(*payload_modulus, *delta, number_of_inputs,
                                    repetitions, samples);

  *d_lwe_ct_out_array = (uint64_t *)cuda_malloc_async(
      (output_lwe_dimension + 1) * number_of_inputs * sizeof(uint64_t), stream,
      gpu_index);
  *d_lwe_ct_in_array = (uint64_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * number_of_inputs * repetitions * samples *
          sizeof(uint64_t),
      stream, gpu_index);
  uint64_t *lwe_ct_in_array =
      (uint64_t *)malloc((input_lwe_dimension + 1) * number_of_inputs *
                         repetitions * samples * sizeof(uint64_t));
  // Create the input/output ciphertexts
  for (int r = 0; r < repetitions; r++) {
    uint64_t *lwe_sk_in =
        *lwe_sk_in_array + (ptrdiff_t)(r * input_lwe_dimension);
    for (int s = 0; s < samples; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = (*plaintexts)[r * samples * number_of_inputs +
                                           s * number_of_inputs + i];
        uint64_t *lwe_ct_in =
            lwe_ct_in_array + (ptrdiff_t)((r * samples * number_of_inputs +
                                           s * number_of_inputs + i) *
                                          (input_lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk_in, lwe_ct_in, plaintext, input_lwe_dimension,
            lwe_modular_variance, *csprng, &CONCRETE_CSPRNG_VTABLE);
      }
    }
  }
  cuda_memcpy_async_to_gpu(*d_lwe_ct_in_array, lwe_ct_in_array,
                           repetitions * samples * number_of_inputs *
                               (input_lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);
  cuda_synchronize_stream(stream);
  free(lwe_ct_in_array);
}

void keyswitch_teardown(cudaStream_t *stream, Csprng *csprng,
                        uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
                        uint64_t *d_ksk_array, uint64_t *plaintexts,
                        uint64_t *d_lwe_ct_in_array,
                        uint64_t *d_lwe_ct_out_array, int gpu_index) {
  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(csprng);
  free(csprng);
  free(lwe_sk_in_array);
  free(lwe_sk_out_array);
  free(plaintexts);

  cuda_drop_async(d_ksk_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_out_array, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

void linear_algebra_setup(cudaStream_t *stream, Csprng **csprng,
                          uint64_t **lwe_sk_array, uint64_t **d_lwe_in_1_ct,
                          uint64_t **d_lwe_in_2_ct, uint64_t **d_lwe_out_ct,
                          uint64_t **lwe_in_1_ct, uint64_t **lwe_in_2_ct,
                          uint64_t **lwe_out_ct, uint64_t **plaintexts_1,
                          uint64_t **plaintexts_2, uint64_t **d_plaintexts_2,
                          uint64_t **d_cleartext_2, int lwe_dimension,
                          double noise_variance, int payload_modulus,
                          uint64_t delta, int number_of_inputs, int repetitions,
                          int samples, int gpu_index) {

  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  // Generate the keys
  generate_lwe_secret_keys(lwe_sk_array, lwe_dimension, *csprng, repetitions);
  *plaintexts_1 = generate_plaintexts(payload_modulus, delta, number_of_inputs,
                                      repetitions, samples);
  *plaintexts_2 = generate_plaintexts(payload_modulus, delta, number_of_inputs,
                                      repetitions, samples);

  *lwe_in_1_ct = (uint64_t *)malloc(repetitions * samples * number_of_inputs *
                                    (lwe_dimension + 1) * sizeof(uint64_t));
  *lwe_in_2_ct = (uint64_t *)malloc(repetitions * samples * number_of_inputs *
                                    (lwe_dimension + 1) * sizeof(uint64_t));
  uint64_t *cleartext_2 = (uint64_t *)malloc(
      repetitions * samples * number_of_inputs * sizeof(uint64_t));
  *lwe_out_ct = (uint64_t *)malloc(number_of_inputs * (lwe_dimension + 1) *
                                   sizeof(uint64_t));

  // Create the input/output ciphertexts
  for (int r = 0; r < repetitions; r++) {
    uint64_t *lwe_sk = *lwe_sk_array + (ptrdiff_t)(r * lwe_dimension);
    for (int s = 0; s < samples; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext_1 = (*plaintexts_1)[r * samples * number_of_inputs +
                                               s * number_of_inputs + i];
        uint64_t plaintext_2 = (*plaintexts_2)[r * samples * number_of_inputs +
                                               s * number_of_inputs + i];
        uint64_t *lwe_1_in =
            (*lwe_in_1_ct) + (ptrdiff_t)((r * samples * number_of_inputs +
                                          s * number_of_inputs + i) *
                                         (lwe_dimension + 1));
        uint64_t *lwe_2_in =
            (*lwe_in_2_ct) + (ptrdiff_t)((r * samples * number_of_inputs +
                                          s * number_of_inputs + i) *
                                         (lwe_dimension + 1));

        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk, lwe_1_in, plaintext_1, lwe_dimension, noise_variance,
            *csprng, &CONCRETE_CSPRNG_VTABLE);
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk, lwe_2_in, plaintext_2, lwe_dimension, noise_variance,
            *csprng, &CONCRETE_CSPRNG_VTABLE);
        cleartext_2[r * samples * number_of_inputs + s * number_of_inputs + i] =
            plaintext_2 / delta;
      }
    }
  }

  // Initialize and copy things in/to the device
  *d_lwe_in_1_ct =
      (uint64_t *)cuda_malloc_async(repetitions * samples * number_of_inputs *
                                        (lwe_dimension + 1) * sizeof(uint64_t),
                                    stream, gpu_index);
  *d_lwe_in_2_ct =
      (uint64_t *)cuda_malloc_async(repetitions * samples * number_of_inputs *
                                        (lwe_dimension + 1) * sizeof(uint64_t),
                                    stream, gpu_index);
  *d_plaintexts_2 = (uint64_t *)cuda_malloc_async(
      repetitions * samples * number_of_inputs * sizeof(uint64_t), stream,
      gpu_index);
  *d_cleartext_2 = (uint64_t *)cuda_malloc_async(
      repetitions * samples * number_of_inputs * sizeof(uint64_t), stream,
      gpu_index);
  *d_lwe_out_ct = (uint64_t *)cuda_malloc_async(
      number_of_inputs * (lwe_dimension + 1) * sizeof(uint64_t), stream,
      gpu_index);

  cuda_memcpy_async_to_gpu(*d_lwe_in_1_ct, *lwe_in_1_ct,
                           repetitions * samples * number_of_inputs *
                               (lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);
  cuda_memcpy_async_to_gpu(*d_lwe_in_2_ct, *lwe_in_2_ct,
                           repetitions * samples * number_of_inputs *
                               (lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);
  cuda_memcpy_async_to_gpu(*d_plaintexts_2, *plaintexts_2,
                           repetitions * samples * number_of_inputs *
                               sizeof(uint64_t),
                           stream, gpu_index);
  cuda_memcpy_async_to_gpu(*d_cleartext_2, cleartext_2,
                           repetitions * samples * number_of_inputs *
                               sizeof(uint64_t),
                           stream, gpu_index);

  cuda_synchronize_stream(stream);
  free(cleartext_2);
}

void linear_algebra_teardown(cudaStream_t *stream, Csprng **csprng,
                             uint64_t **lwe_sk_array, uint64_t **d_lwe_in_1_ct,
                             uint64_t **d_lwe_in_2_ct, uint64_t **d_lwe_out_ct,
                             uint64_t **lwe_in_1_ct, uint64_t **lwe_in_2_ct,
                             uint64_t **lwe_out_ct, uint64_t **plaintexts_1,
                             uint64_t **plaintexts_2, uint64_t **d_plaintexts_2,
                             uint64_t **d_cleartext_2, int gpu_index) {

  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(*csprng);
  free(*csprng);
  free(*lwe_out_ct);
  free(*lwe_sk_array);
  free(*plaintexts_1);
  free(*plaintexts_2);
  free(*lwe_in_1_ct);
  free(*lwe_in_2_ct);

  cuda_drop_async(*d_lwe_in_1_ct, stream, gpu_index);
  cuda_drop_async(*d_lwe_in_2_ct, stream, gpu_index);
  cuda_drop_async(*d_plaintexts_2, stream, gpu_index);
  cuda_drop_async(*d_cleartext_2, stream, gpu_index);
  cuda_drop_async(*d_lwe_out_ct, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

void bit_extraction_setup(
    cudaStream_t *stream, Csprng **csprng, uint64_t **lwe_sk_in_array,
    uint64_t **lwe_sk_out_array, double **d_fourier_bsk_array,
    uint64_t **d_ksk_array, uint64_t **plaintexts, uint64_t **d_lwe_ct_in_array,
    uint64_t **d_lwe_ct_out_array, int8_t **bit_extract_buffer,
    int lwe_dimension, int glwe_dimension, int polynomial_size,
    double lwe_modular_variance, double glwe_modular_variance, int ks_base_log,
    int ks_level, int pbs_base_log, int pbs_level,
    int number_of_bits_of_message_including_padding,
    int number_of_bits_to_extract, int *delta_log, uint64_t *delta,
    int number_of_inputs, int repetitions, int samples, int gpu_index) {
  *delta_log = 64 - number_of_bits_of_message_including_padding;
  *delta = (uint64_t)(1) << *delta_log;

  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  int input_lwe_dimension = glwe_dimension * polynomial_size;
  int output_lwe_dimension = lwe_dimension;
  // Generate the keys
  generate_lwe_secret_keys(lwe_sk_in_array, input_lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_secret_keys(lwe_sk_out_array, output_lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_keyswitch_keys(stream, gpu_index, d_ksk_array, *lwe_sk_in_array,
                              *lwe_sk_out_array, input_lwe_dimension,
                              output_lwe_dimension, ks_level, ks_base_log,
                              *csprng, lwe_modular_variance, repetitions);

  generate_lwe_bootstrap_keys(
      stream, gpu_index, d_fourier_bsk_array, *lwe_sk_out_array,
      *lwe_sk_in_array, lwe_dimension, glwe_dimension, polynomial_size,
      pbs_level, pbs_base_log, *csprng, glwe_modular_variance, repetitions);
  *plaintexts =
      generate_plaintexts(number_of_bits_of_message_including_padding, *delta,
                          number_of_inputs, repetitions, samples);

  *d_lwe_ct_out_array = (uint64_t *)cuda_malloc_async(
      (output_lwe_dimension + 1) * number_of_bits_to_extract *
          number_of_inputs * sizeof(uint64_t),
      stream, gpu_index);

  *d_lwe_ct_in_array = (uint64_t *)cuda_malloc_async(
      (input_lwe_dimension + 1) * number_of_inputs * repetitions * samples *
          sizeof(uint64_t),
      stream, gpu_index);

  uint64_t *lwe_ct_in_array =
      (uint64_t *)malloc(repetitions * samples * (input_lwe_dimension + 1) *
                         number_of_inputs * sizeof(uint64_t));

  // Create the input ciphertexts
  for (int r = 0; r < repetitions; r++) {
    uint64_t *lwe_sk_in =
        *lwe_sk_in_array + (ptrdiff_t)(r * input_lwe_dimension);
    for (int s = 0; s < samples; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = (*plaintexts)[r * samples * number_of_inputs +
                                           s * number_of_inputs + i];
        uint64_t *lwe_ct_in =
            lwe_ct_in_array + (ptrdiff_t)((r * samples * number_of_inputs +
                                           s * number_of_inputs + i) *
                                          (input_lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk_in, lwe_ct_in, plaintext, input_lwe_dimension,
            lwe_modular_variance, *csprng, &CONCRETE_CSPRNG_VTABLE);
      }
    }
  }
  cuda_memcpy_async_to_gpu(*d_lwe_ct_in_array, lwe_ct_in_array,
                           repetitions * samples * number_of_inputs *
                               (input_lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);
  // Execute scratch
  scratch_cuda_extract_bits_64(stream, gpu_index, bit_extract_buffer,
                               glwe_dimension, lwe_dimension, polynomial_size,
                               pbs_level, number_of_inputs,
                               cuda_get_max_shared_memory(gpu_index), true);

  cuda_synchronize_stream(stream);
  free(lwe_ct_in_array);
}

void bit_extraction_teardown(cudaStream_t *stream, Csprng *csprng,
                             uint64_t *lwe_sk_in_array,
                             uint64_t *lwe_sk_out_array,
                             double *d_fourier_bsk_array, uint64_t *d_ksk_array,
                             uint64_t *plaintexts, uint64_t *d_lwe_ct_in_array,
                             uint64_t *d_lwe_ct_out_array,
                             int8_t *bit_extract_buffer, int gpu_index) {
  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(csprng);
  free(csprng);
  free(lwe_sk_in_array);
  free(lwe_sk_out_array);
  free(plaintexts);

  cleanup_cuda_extract_bits(stream, gpu_index, &bit_extract_buffer);
  cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
  cuda_drop_async(d_ksk_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_out_array, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

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
    int samples, int gpu_index) {

  *delta_log = 60;

  *delta = (uint64_t)(1) << *delta_log;
  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  // Generate the keys
  generate_lwe_secret_keys(lwe_sk_in_array, lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_secret_keys(lwe_sk_out_array, glwe_dimension * polynomial_size,
                           *csprng, repetitions);
  generate_lwe_bootstrap_keys(
      stream, gpu_index, d_fourier_bsk_array, *lwe_sk_in_array,
      *lwe_sk_out_array, lwe_dimension, glwe_dimension, polynomial_size,
      pbs_level, pbs_base_log, *csprng, glwe_modular_variance, repetitions);
  generate_lwe_private_functional_keyswitch_key_lists(
      stream, gpu_index, d_pksk_array, *lwe_sk_out_array, *lwe_sk_out_array,
      glwe_dimension * polynomial_size, glwe_dimension, polynomial_size,
      pksk_level, pksk_base_log, *csprng, lwe_modular_variance, repetitions);
  *plaintexts =
      generate_plaintexts(number_of_bits_of_message_including_padding, *delta,
                          number_of_inputs, repetitions, samples);

  *d_ggsw_ct_out_array = (uint64_t *)cuda_malloc_async(
      repetitions * samples * number_of_inputs * ggsw_size * sizeof(uint64_t),
      stream, gpu_index);

  *d_lwe_ct_in_array =
      (uint64_t *)cuda_malloc_async(repetitions * samples * number_of_inputs *
                                        (lwe_dimension + 1) * sizeof(uint64_t),
                                    stream, gpu_index);

  uint64_t *lwe_ct_in_array =
      (uint64_t *)malloc(repetitions * samples * number_of_inputs *
                         (lwe_dimension + 1) * sizeof(uint64_t));
  // Create the input ciphertexts
  for (int r = 0; r < repetitions; r++) {
    uint64_t *lwe_sk_in = *lwe_sk_in_array + (ptrdiff_t)(r * lwe_dimension);
    for (int s = 0; s < samples; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        uint64_t plaintext = (*plaintexts)[r * samples * number_of_inputs +
                                           s * number_of_inputs + i];
        uint64_t *lwe_ct_in =
            lwe_ct_in_array + (ptrdiff_t)((r * samples * number_of_inputs +
                                           s * number_of_inputs + i) *
                                          (lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk_in, lwe_ct_in, plaintext, lwe_dimension,
            lwe_modular_variance, *csprng, &CONCRETE_CSPRNG_VTABLE);
      }
    }
  }
  cuda_memcpy_async_to_gpu(*d_lwe_ct_in_array, lwe_ct_in_array,
                           repetitions * samples * number_of_inputs *
                               (lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);

  // Execute cbs scratch
  scratch_cuda_circuit_bootstrap_64(
      stream, gpu_index, cbs_buffer, glwe_dimension, lwe_dimension,
      polynomial_size, cbs_level, number_of_inputs,
      cuda_get_max_shared_memory(gpu_index), true);

  // Build LUT vector indexes
  uint64_t *h_lut_vector_indexes =
      (uint64_t *)malloc(number_of_inputs * cbs_level * sizeof(uint64_t));
  for (int index = 0; index < cbs_level * number_of_inputs; index++) {
    h_lut_vector_indexes[index] = index % cbs_level;
  }
  *d_lut_vector_indexes = (uint64_t *)cuda_malloc_async(
      number_of_inputs * cbs_level * sizeof(uint64_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(*d_lut_vector_indexes, h_lut_vector_indexes,
                           number_of_inputs * cbs_level * sizeof(uint64_t),
                           stream, gpu_index);
  cuda_synchronize_stream(stream);
  free(h_lut_vector_indexes);
  free(lwe_ct_in_array);
}

void circuit_bootstrap_teardown(
    cudaStream_t *stream, Csprng *csprng, uint64_t *lwe_sk_in_array,
    uint64_t *lwe_sk_out_array, double *d_fourier_bsk_array,
    uint64_t *d_pksk_array, uint64_t *plaintexts, uint64_t *d_lwe_ct_in_array,
    uint64_t *d_lut_vector_indexes, uint64_t *d_ggsw_ct_out_array,
    int8_t *cbs_buffer, int gpu_index) {

  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(csprng);
  free(csprng);
  free(lwe_sk_in_array);
  free(lwe_sk_out_array);
  free(plaintexts);

  cleanup_cuda_circuit_bootstrap(stream, gpu_index, &cbs_buffer);
  cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
  cuda_drop_async(d_pksk_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
  cuda_drop_async(d_ggsw_ct_out_array, stream, gpu_index);
  cuda_drop_async(d_lut_vector_indexes, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

void cmux_tree_setup(cudaStream_t *stream, Csprng **csprng, uint64_t **glwe_sk,
                     uint64_t **d_lut_identity, uint64_t **plaintexts,
                     uint64_t **d_ggsw_bit_array, int8_t **cmux_tree_buffer,
                     uint64_t **d_glwe_out, int glwe_dimension,
                     int polynomial_size, int base_log, int level_count,
                     double glwe_modular_variance, int r_lut, int tau,
                     uint64_t delta_log, int repetitions, int samples,
                     int gpu_index) {
  int ggsw_size = polynomial_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count;
  int glwe_size = (glwe_dimension + 1) * polynomial_size;

  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  // Generate the keys
  generate_glwe_secret_keys(glwe_sk, glwe_dimension, polynomial_size, *csprng,
                            repetitions);
  *plaintexts = generate_plaintexts(r_lut, 1, 1, repetitions, samples);

  // Create the LUT
  int num_lut = (1 << r_lut);
  *d_lut_identity = (uint64_t *)cuda_malloc_async(
      polynomial_size * num_lut * tau * sizeof(uint64_t), stream, gpu_index);
  uint64_t *lut_cmux_tree_identity =
      generate_identity_lut_cmux_tree(polynomial_size, num_lut, tau, delta_log);

  // Encrypt one bit per GGSW
  uint64_t *ggsw_bit_array = (uint64_t *)malloc(repetitions * samples * r_lut *
                                                ggsw_size * sizeof(uint64_t));
  for (int r = 0; r < repetitions; r++) {
    for (int s = 0; s < samples; s++) {
      uint64_t witness = (*plaintexts)[r * samples + s];

      // Instantiate the GGSW m^tree ciphertexts
      // We need r GGSW ciphertexts
      // Bit decomposition of the value from MSB to LSB
      uint64_t *bit_array = bit_decompose_value(witness, r_lut);
      for (int i = 0; i < r_lut; i++) {
        uint64_t *ggsw_slice =
            ggsw_bit_array +
            (ptrdiff_t)((r * samples * r_lut + s * r_lut + i) * ggsw_size);
        concrete_cpu_encrypt_ggsw_ciphertext_u64(
            *glwe_sk, ggsw_slice, bit_array[i], glwe_dimension, polynomial_size,
            level_count, base_log, glwe_modular_variance, *csprng,
            &CONCRETE_CSPRNG_VTABLE);
      }
      free(bit_array);
    }
  }
  // Allocate and copy things to the device
  *d_glwe_out = (uint64_t *)cuda_malloc_async(
      tau * glwe_size * sizeof(uint64_t), stream, gpu_index);
  *d_ggsw_bit_array = (uint64_t *)cuda_malloc_async(
      repetitions * samples * r_lut * ggsw_size * sizeof(uint64_t), stream,
      gpu_index);
  cuda_memcpy_async_to_gpu(*d_lut_identity, lut_cmux_tree_identity,
                           polynomial_size * num_lut * tau * sizeof(uint64_t),
                           stream, gpu_index);
  cuda_memcpy_async_to_gpu(*d_ggsw_bit_array, ggsw_bit_array,
                           repetitions * samples * r_lut * ggsw_size *
                               sizeof(uint64_t),
                           stream, gpu_index);

  scratch_cuda_cmux_tree_64(stream, gpu_index, cmux_tree_buffer, glwe_dimension,
                            polynomial_size, level_count, r_lut, tau,
                            cuda_get_max_shared_memory(gpu_index), true);
  cuda_synchronize_stream(stream);
  free(lut_cmux_tree_identity);
  free(ggsw_bit_array);
}
void cmux_tree_teardown(cudaStream_t *stream, Csprng **csprng,
                        uint64_t **glwe_sk, uint64_t **d_lut_identity,
                        uint64_t **plaintexts, uint64_t **d_ggsw_bit_array,
                        int8_t **cmux_tree_buffer, uint64_t **d_glwe_out,
                        int gpu_index) {
  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(*csprng);
  free(*plaintexts);
  free(*csprng);
  free(*glwe_sk);

  cleanup_cuda_cmux_tree(stream, gpu_index, cmux_tree_buffer);
  cuda_drop_async(*d_lut_identity, stream, gpu_index);
  cuda_drop_async(*d_ggsw_bit_array, stream, gpu_index);
  cuda_drop_async(*d_glwe_out, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

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
                   int cbs_level, int p, int *delta_log, int *cbs_delta_log,
                   int *delta_log_lut, uint64_t *delta, int tau,
                   int repetitions, int samples, int gpu_index) {

  int input_lwe_dimension = glwe_dimension * polynomial_size;
  *delta_log = 64 - p;
  *delta_log_lut = *delta_log;
  *delta = (uint64_t)(1) << *delta_log;
  // Create a Csprng
  *csprng =
      (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  uint8_t seed[16] = {(uint8_t)0};
  concrete_cpu_construct_concrete_csprng(
      *csprng, Uint128{.little_endian_bytes = {*seed}});

  // Generate the keys
  generate_lwe_secret_keys(lwe_sk_in_array, input_lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_secret_keys(lwe_sk_out_array, lwe_dimension, *csprng,
                           repetitions);
  generate_lwe_keyswitch_keys(stream, gpu_index, d_ksk_array, *lwe_sk_in_array,
                              *lwe_sk_out_array, input_lwe_dimension,
                              lwe_dimension, ks_level, ks_base_log, *csprng,
                              lwe_modular_variance, repetitions);
  generate_lwe_bootstrap_keys(
      stream, gpu_index, d_fourier_bsk_array, *lwe_sk_out_array,
      *lwe_sk_in_array, lwe_dimension, glwe_dimension, polynomial_size,
      pbs_level, pbs_base_log, *csprng, glwe_modular_variance, repetitions);
  generate_lwe_private_functional_keyswitch_key_lists(
      stream, gpu_index, d_pksk_array, *lwe_sk_in_array, *lwe_sk_in_array,
      input_lwe_dimension, glwe_dimension, polynomial_size, pksk_level,
      pksk_base_log, *csprng, lwe_modular_variance, repetitions);
  *plaintexts = generate_plaintexts(p, *delta, tau, repetitions, samples);

  // LUT creation
  int lut_size = polynomial_size;
  int lut_num = tau << (tau * p - (int)log2(polynomial_size)); // r

  uint64_t *big_lut = (uint64_t *)malloc(lut_num * lut_size * sizeof(uint64_t));
  for (int t = tau - 1; t >= 0; t--) {
    uint64_t *small_lut = big_lut + (ptrdiff_t)(t * (1 << (tau * p)));
    for (uint64_t value = 0; value < (uint64_t)(1 << (tau * p)); value++) {
      int nbits = t * p;
      uint64_t x = (value >> nbits) & (uint64_t)((1 << p) - 1);
      small_lut[value] =
          ((x % (uint64_t)(1 << (64 - *delta_log))) << *delta_log_lut);
    }
  }
  *d_lut_vector = (uint64_t *)cuda_malloc_async(
      lut_num * lut_size * sizeof(uint64_t), stream, gpu_index);
  cuda_memcpy_async_to_gpu(*d_lut_vector, big_lut,
                           lut_num * lut_size * sizeof(uint64_t), stream,
                           gpu_index);
  // Allocate input
  *d_lwe_ct_in_array = (uint64_t *)cuda_malloc_async(
      repetitions * samples * (input_lwe_dimension + 1) * tau *
          sizeof(uint64_t),
      stream, gpu_index);
  // Allocate output
  *d_lwe_ct_out_array = (uint64_t *)cuda_malloc_async(
      repetitions * samples * (input_lwe_dimension + 1) * tau *
          sizeof(uint64_t),
      stream, gpu_index);
  uint64_t *lwe_ct_in_array =
      (uint64_t *)malloc(repetitions * samples * (input_lwe_dimension + 1) *
                         tau * sizeof(uint64_t));
  // Create the input ciphertexts
  for (int r = 0; r < repetitions; r++) {
    uint64_t *lwe_sk_in =
        *lwe_sk_in_array + (ptrdiff_t)(r * input_lwe_dimension);
    for (int s = 0; s < samples; s++) {
      for (int i = 0; i < tau; i++) {
        uint64_t plaintext = (*plaintexts)[r * samples * tau + s * tau + i];
        uint64_t *lwe_ct_in =
            lwe_ct_in_array + (ptrdiff_t)((r * samples * tau + s * tau + i) *
                                          (input_lwe_dimension + 1));
        concrete_cpu_encrypt_lwe_ciphertext_u64(
            lwe_sk_in, lwe_ct_in, plaintext, input_lwe_dimension,
            lwe_modular_variance, *csprng, &CONCRETE_CSPRNG_VTABLE);
      }
    }
  }
  cuda_memcpy_async_to_gpu(*d_lwe_ct_in_array, lwe_ct_in_array,
                           repetitions * samples * tau *
                               (input_lwe_dimension + 1) * sizeof(uint64_t),
                           stream, gpu_index);
  // Execute scratch
  scratch_cuda_wop_pbs_64(stream, gpu_index, wop_pbs_buffer,
                          (uint32_t *)delta_log, (uint32_t *)cbs_delta_log,
                          glwe_dimension, lwe_dimension, polynomial_size,
                          cbs_level, pbs_level, p, p, tau,
                          cuda_get_max_shared_memory(gpu_index), true);

  cuda_synchronize_stream(stream);
  free(lwe_ct_in_array);
  free(big_lut);
}

void wop_pbs_teardown(cudaStream_t *stream, Csprng *csprng,
                      uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
                      uint64_t *d_ksk_array, double *d_fourier_bsk_array,
                      uint64_t *d_pksk_array, uint64_t *plaintexts,
                      uint64_t *d_lwe_ct_in_array, uint64_t *d_lut_vector,
                      uint64_t *d_lwe_ct_out_array, int8_t *wop_pbs_buffer,
                      int gpu_index) {
  cuda_synchronize_stream(stream);

  concrete_cpu_destroy_concrete_csprng(csprng);
  free(csprng);
  free(lwe_sk_in_array);
  free(lwe_sk_out_array);
  free(plaintexts);

  cleanup_cuda_wop_pbs(stream, gpu_index, &wop_pbs_buffer);
  cuda_drop_async(d_fourier_bsk_array, stream, gpu_index);
  cuda_drop_async(d_ksk_array, stream, gpu_index);
  cuda_drop_async(d_pksk_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_in_array, stream, gpu_index);
  cuda_drop_async(d_lwe_ct_out_array, stream, gpu_index);
  cuda_drop_async(d_lut_vector, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}

void fft_setup(cudaStream_t *stream, double **_poly1, double **_poly2,
               double2 **_h_cpoly1, double2 **_h_cpoly2, double2 **_d_cpoly1,
               double2 **_d_cpoly2, size_t polynomial_size, int samples,
               int gpu_index) {

  auto &poly1 = *_poly1;
  auto &poly2 = *_poly2;
  auto &h_cpoly1 = *_h_cpoly1;
  auto &h_cpoly2 = *_h_cpoly2;
  auto &d_cpoly1 = *_d_cpoly1;
  auto &d_cpoly2 = *_d_cpoly2;

  poly1 = (double *)malloc(polynomial_size * samples * sizeof(double));
  poly2 = (double *)malloc(polynomial_size * samples * sizeof(double));
  h_cpoly1 = (double2 *)malloc(polynomial_size / 2 * samples * sizeof(double2));
  h_cpoly2 = (double2 *)malloc(polynomial_size / 2 * samples * sizeof(double2));
  d_cpoly1 = (double2 *)cuda_malloc_async(
      polynomial_size / 2 * samples * sizeof(double2), stream, gpu_index);
  d_cpoly2 = (double2 *)cuda_malloc_async(
      polynomial_size / 2 * samples * sizeof(double2), stream, gpu_index);

  double lower_bound = -1;
  double upper_bound = 1;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;
  // Fill test data with random values
  for (size_t i = 0; i < polynomial_size * samples; i++) {
    poly1[i] = unif(re);
    poly2[i] = unif(re);
  }

  // prepare data for device
  // compress
  for (size_t p = 0; p < (size_t)samples; p++) {
    auto left_cpoly = &h_cpoly1[p * polynomial_size / 2];
    auto right_cpoly = &h_cpoly2[p * polynomial_size / 2];
    auto left = &poly1[p * polynomial_size];
    auto right = &poly2[p * polynomial_size];
    for (std::size_t i = 0; i < polynomial_size / 2; ++i) {
      left_cpoly[i].x = left[i];
      left_cpoly[i].y = left[i + polynomial_size / 2];

      right_cpoly[i].x = right[i];
      right_cpoly[i].y = right[i + polynomial_size / 2];
    }
  }

  // copy memory cpu->gpu
  cuda_memcpy_async_to_gpu(d_cpoly1, h_cpoly1,
                           polynomial_size / 2 * samples * sizeof(double2),
                           stream, gpu_index);
  cuda_memcpy_async_to_gpu(d_cpoly2, h_cpoly2,
                           polynomial_size / 2 * samples * sizeof(double2),
                           stream, gpu_index);
  cuda_synchronize_stream(stream);
}

void fft_teardown(cudaStream_t *stream, double *poly1, double *poly2,
                  double2 *h_cpoly1, double2 *h_cpoly2, double2 *d_cpoly1,
                  double2 *d_cpoly2, int gpu_index) {
  cuda_synchronize_stream(stream);

  free(poly1);
  free(poly2);
  free(h_cpoly1);
  free(h_cpoly2);

  cuda_drop_async(d_cpoly1, stream, gpu_index);
  cuda_drop_async(d_cpoly2, stream, gpu_index);
  cuda_synchronize_stream(stream);
  cuda_destroy_stream(stream, gpu_index);
}
