#include "utils.h"
#include "../include/bootstrap.h"
#include "../include/device.h"
#include "concrete-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <random>

// For each sample and repetition, create a plaintext
// The payload_modulus is the message modulus times the carry modulus
// (so the total message modulus)
uint64_t *generate_plaintexts(uint64_t payload_modulus, uint64_t delta,
                              int number_of_inputs, const unsigned repetitions, const unsigned
                              samples) {
  uint64_t *plaintext_array = (uint64_t *)malloc(
      repetitions * samples * number_of_inputs * sizeof(uint64_t));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned long long> dis(
      std::numeric_limits<std::uint64_t>::min(),
      std::numeric_limits<std::uint64_t>::max());
  for (uint r = 0; r < repetitions; r++) {
    for (uint s = 0; s < samples; s++) {
      for (int i = 0; i < number_of_inputs; i++) {
        plaintext_array[r * samples * number_of_inputs + s * number_of_inputs +
                        i] = (dis(gen) % payload_modulus) * delta;
      }
    }
  }
  return plaintext_array;
}

// Decompose value in r bits
// Bit decomposition of the value from MSB to LSB
uint64_t *bit_decompose_value(uint64_t value, int r) {
  uint64_t *bit_array = (uint64_t *)malloc(r * sizeof(uint64_t));

  uint64_t x = value;
  for (int i = 0; i < r; i++) {
    bit_array[i] = x & 1;
    x >>= 1;
  }
  return bit_array;
}

uint64_t *generate_identity_lut_pbs(int polynomial_size, int glwe_dimension,
                                    int message_modulus, int carry_modulus,
                                    std::function<uint64_t(uint64_t)> func) {
  // Modulus of the msg contained in the msg bits and operations buffer
  uint64_t modulus_sup = message_modulus * carry_modulus;

  // N/(p/2) = size of each block
  uint64_t box_size = polynomial_size / modulus_sup;

  // Value of the shift we multiply our messages by
  uint64_t delta = ((uint64_t)1 << 63) / (uint64_t)(modulus_sup);

  // Create the plaintext lut_pbs
  uint64_t *plaintext_lut_pbs =
      (uint64_t *)malloc(polynomial_size * sizeof(uint64_t));

  // This plaintext_lut_pbs extracts the carry bits
  for (uint64_t i = 0; i < modulus_sup; i++) {
    uint64_t index = i * box_size;
    for (uint64_t j = index; j < index + box_size; j++) {
      plaintext_lut_pbs[j] = func(i) * delta;
    }
  }

  uint64_t half_box_size = box_size / 2;

  // Negate the first half_box_size coefficients
  for (uint64_t i = 0; i < half_box_size; i++) {
    plaintext_lut_pbs[i] = -plaintext_lut_pbs[i];
  }

  // Rotate the plaintext_lut_pbs
  std::rotate(plaintext_lut_pbs, plaintext_lut_pbs + half_box_size,
              plaintext_lut_pbs + polynomial_size);

  // Create the GLWE lut_pbs
  uint64_t *lut_pbs = (uint64_t *)malloc(
      polynomial_size * (glwe_dimension + 1) * sizeof(uint64_t));
  for (int i = 0; i < polynomial_size * glwe_dimension; i++) {
    lut_pbs[i] = 0;
  }
  for (int i = 0; i < polynomial_size; i++) {
    int glwe_index = glwe_dimension * polynomial_size + i;
    lut_pbs[glwe_index] = plaintext_lut_pbs[i];
  }

  free(plaintext_lut_pbs);
  return lut_pbs;
}

uint64_t *generate_identity_lut_cmux_tree(int polynomial_size, int num_lut,
                                          int tau, int delta_log) {

  // Create the plaintext lut_pbs
  uint64_t *plaintext_lut_cmux_tree =
      (uint64_t *)malloc(num_lut * tau * polynomial_size * sizeof(uint64_t));

  // This plaintext_lut_cmux_tree extracts the carry bits
  for (int tree = 0; tree < tau; tree++)
    for (int i = 0; i < num_lut; i++) {
      uint64_t *plaintext_lut_slice = plaintext_lut_cmux_tree +
                                      i * polynomial_size +
                                      tree * num_lut * polynomial_size;
      uint64_t coeff = (((uint64_t)(i + tree) % (1 << (64 - delta_log))))
                       << delta_log;
      for (int p = 0; p < polynomial_size; p++)
        plaintext_lut_slice[p] = coeff;
    }

  return plaintext_lut_cmux_tree;
}

// Generate repetitions LWE secret keys
void generate_lwe_secret_keys(uint64_t **lwe_sk_array, int lwe_dimension,
                              Csprng *csprng, const unsigned repetitions) {
  int lwe_sk_array_size = lwe_dimension * repetitions;
  *lwe_sk_array = (uint64_t *)malloc(lwe_sk_array_size * sizeof(uint64_t));
  int shift = 0;
  for (uint r = 0; r < repetitions; r++) {
    // Generate the lwe secret key for each repetition
    concrete_cpu_init_secret_key_u64(*lwe_sk_array + (ptrdiff_t)(shift),
                                     lwe_dimension, csprng,
                                     &CONCRETE_CSPRNG_VTABLE);
    shift += lwe_dimension;
  }
}

// Generate repetitions GLWE secret keys
void generate_glwe_secret_keys(uint64_t **glwe_sk_array, int glwe_dimension,
                               int polynomial_size, Csprng *csprng, const unsigned repetitions) {
  int glwe_sk_array_size = glwe_dimension * polynomial_size * repetitions;
  *glwe_sk_array = (uint64_t *)malloc(glwe_sk_array_size * sizeof(uint64_t));
  int shift = 0;
  for (uint r = 0; r < repetitions; r++) {
    // Generate the lwe secret key for each repetition
    concrete_cpu_init_secret_key_u64(*glwe_sk_array + (ptrdiff_t)(shift),
                                     glwe_dimension * polynomial_size, csprng,
                                     &CONCRETE_CSPRNG_VTABLE);
    shift += glwe_dimension * polynomial_size;
  }
}

// Generate repetitions LWE bootstrap keys
void generate_lwe_bootstrap_keys(cudaStream_t *stream, int gpu_index,
                                 double **d_fourier_bsk_array,
                                 uint64_t *lwe_sk_in_array,
                                 uint64_t *lwe_sk_out_array, int lwe_dimension,
                                 int glwe_dimension, int polynomial_size,
                                 int pbs_level, int pbs_base_log,
                                 Csprng *csprng, double variance, const unsigned repetitions) {
  void *v_stream = (void *)stream;
  int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                 polynomial_size * (lwe_dimension + 1);
  int bsk_array_size = bsk_size * repetitions;

  uint64_t *bsk_array = (uint64_t *)malloc(bsk_array_size * sizeof(uint64_t));
  *d_fourier_bsk_array = (double *)cuda_malloc_async(
      bsk_array_size * sizeof(double), stream, gpu_index);
  int shift_in = 0;
  int shift_out = 0;
  int shift_bsk = 0;

  for (uint r = 0; r < repetitions; r++) {
    // Generate the bootstrap key for each repetition
    concrete_cpu_init_lwe_bootstrap_key_u64(
        bsk_array + (ptrdiff_t)(shift_bsk),
        lwe_sk_in_array + (ptrdiff_t)(shift_in),
        lwe_sk_out_array + (ptrdiff_t)(shift_out), lwe_dimension,
        polynomial_size, glwe_dimension, pbs_level, pbs_base_log, variance,
        Parallelism(1), csprng, &CONCRETE_CSPRNG_VTABLE);
    cuda_synchronize_stream(v_stream);
    double *d_fourier_bsk = *d_fourier_bsk_array + (ptrdiff_t)(shift_bsk);
    uint64_t *bsk = bsk_array + (ptrdiff_t)(shift_bsk);
    cuda_synchronize_stream(v_stream);
    cuda_convert_lwe_bootstrap_key_64(
        (void *)(d_fourier_bsk), (void *)(bsk), v_stream, gpu_index,
        lwe_dimension, glwe_dimension, pbs_level, polynomial_size);
    shift_in += lwe_dimension;
    shift_out += glwe_dimension * polynomial_size;
    shift_bsk += bsk_size;
  }
  free(bsk_array);
}

// Generate repetitions keyswitch keys
void generate_lwe_keyswitch_keys(
    cudaStream_t *stream, int gpu_index, uint64_t **d_ksk_array,
    uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
    int input_lwe_dimension, int output_lwe_dimension, int ksk_level,
    int ksk_base_log, Csprng *csprng, double variance, const unsigned repetitions) {

  int ksk_size = ksk_level * (output_lwe_dimension + 1) * input_lwe_dimension;
  int ksk_array_size = ksk_size * repetitions;

  uint64_t *ksk_array = (uint64_t *)malloc(ksk_array_size * sizeof(uint64_t));
  *d_ksk_array = (uint64_t *)cuda_malloc_async(
      ksk_array_size * sizeof(uint64_t), stream, gpu_index);
  int shift_in = 0;
  int shift_out = 0;
  int shift_ksk = 0;

  for (uint r = 0; r < repetitions; r++) {
    // Generate the keyswitch key for each repetition
    concrete_cpu_init_lwe_keyswitch_key_u64(
        ksk_array + (ptrdiff_t)(shift_ksk),
        lwe_sk_in_array + (ptrdiff_t)(shift_in),
        lwe_sk_out_array + (ptrdiff_t)(shift_out), input_lwe_dimension,
        output_lwe_dimension, ksk_level, ksk_base_log, variance, csprng,
        &CONCRETE_CSPRNG_VTABLE);
    uint64_t *d_ksk = *d_ksk_array + (ptrdiff_t)(shift_ksk);
    uint64_t *ksk = ksk_array + (ptrdiff_t)(shift_ksk);
    cuda_memcpy_async_to_gpu(d_ksk, ksk, ksk_size * sizeof(uint64_t), stream,
                             gpu_index);

    shift_in += input_lwe_dimension;
    shift_out += output_lwe_dimension;
    shift_ksk += ksk_size;
  }
  free(ksk_array);
}

// Generate repetitions private functional keyswitch key lists (with (k + 1)
// keys each)
void generate_lwe_private_functional_keyswitch_key_lists(
    cudaStream_t *stream, int gpu_index, uint64_t **d_pksk_array,
    uint64_t *lwe_sk_in_array, uint64_t *lwe_sk_out_array,
    int input_lwe_dimension, int output_glwe_dimension,
    int output_polynomial_size, int pksk_level, int pksk_base_log,
    Csprng *csprng, double variance, const unsigned repetitions) {

  int pksk_list_size = pksk_level * (output_glwe_dimension + 1) *
                       output_polynomial_size * (input_lwe_dimension + 1) *
                       (output_glwe_dimension + 1);
  int pksk_array_size = pksk_list_size * repetitions;

  uint64_t *pksk_array = (uint64_t *)malloc(pksk_array_size * sizeof(uint64_t));
  *d_pksk_array = (uint64_t *)cuda_malloc_async(
      pksk_array_size * sizeof(uint64_t), stream, gpu_index);
  int shift_in = 0;
  int shift_out = 0;
  int shift_pksk_list = 0;

  for (uint r = 0; r < repetitions; r++) {
    // Generate the (k + 1) private functional keyswitch keys for each
    // repetition
    concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
        pksk_array + (ptrdiff_t)(shift_pksk_list),
        lwe_sk_in_array + (ptrdiff_t)(shift_in),
        lwe_sk_out_array + (ptrdiff_t)(shift_out), input_lwe_dimension,
        output_polynomial_size, output_glwe_dimension, pksk_level,
        pksk_base_log, variance, Parallelism(1), csprng,
        &CONCRETE_CSPRNG_VTABLE);
    uint64_t *d_pksk_list = *d_pksk_array + (ptrdiff_t)(shift_pksk_list);
    uint64_t *pksk_list = pksk_array + (ptrdiff_t)(shift_pksk_list);
    cuda_memcpy_async_to_gpu(d_pksk_list, pksk_list,
                             pksk_list_size * sizeof(uint64_t), stream,
                             gpu_index);

    shift_in += input_lwe_dimension;
    shift_out += output_glwe_dimension * output_polynomial_size;
    shift_pksk_list += pksk_list_size;
  }
  free(pksk_array);
}

// The closest number representable by the decomposition can be computed by
// performing the rounding at the appropriate bit.
uint64_t closest_representable(uint64_t input, int level_count, int base_log) {
  // Compute the number of least significant bits which can not be represented
  // by the decomposition
  int non_rep_bit_count = 64 - (level_count * base_log);
  // Generate a mask which captures the non representable bits
  uint64_t one = 1;
  uint64_t non_rep_mask = one << (non_rep_bit_count - 1);
  // Retrieve the non representable bits
  uint64_t non_rep_bits = input & non_rep_mask;
  // Extract the msb of the  non representable bits to perform the rounding
  uint64_t non_rep_msb = non_rep_bits >> (non_rep_bit_count - 1);
  // Remove the non-representable bits and perform the rounding
  uint64_t res = input >> non_rep_bit_count;
  res += non_rep_msb;
  return res << non_rep_bit_count;
}