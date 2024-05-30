//#define ROOT_PROFILING
#define NO_PAPI_PROFILING

#include "bootstrap.h"
#include "bootstrap_optimized.h"
#include "segment.h"
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static uint32_t ilog2(uint32_t val) { return 31 - __builtin_clz(val); }

static inline c64 c64_mul(c64 a, c64 b) {
  c64 ret;

  ret.c[0] = a.c[0] * b.c[0] - a.c[1] * b.c[1];
  ret.c[1] = a.c[0] * b.c[1] + a.c[1] * b.c[0];

  return ret;
}

void memref_bootstrap_lwe_u64_c_optimized(
    uint64_t output_ct[OPT_OUT_SIZE], uint64_t input_ct[OPT_INPUT_SIZE],
    uint64_t tlu[OPT_TLU_SIZE],
    const c64 bootstrap_key[OPT_INPUT_LWE_DIMENSION]
                           [OPT_DECOMPOSITION_LEVEL_COUNT][OPT_GLWE_SIZE]
                           [OPT_GLWE_SIZE][OPT_FOURIER_POLYNOMIAL_SIZE],
    const Fft *fft);

void memref_bootstrap_lwe_u64_c_generic(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dimension, uint32_t polynomial_size,
    uint32_t decomposition_level_count, uint32_t decomposition_base_log,
    uint32_t glwe_dimension, const c64 *bootstrap_key, const Fft *fft) {
  puts("C bootstrap generic");

  if (setup_papi() != 0) {
    exit(1);
  }

  SEGMENT_DECLARE_ROOT(root);
  SEGMENT_DECLARE(intro, &root);
  SEGMENT_DECLARE(blind_rotate, &root);
  SEGMENT_DECLARE(cmux, &blind_rotate);
  SEGMENT_DECLARE(cmux_assignment_start, &cmux);
  SEGMENT_DECLARE(cmux_assignment_end, &cmux);
  SEGMENT_DECLARE(cmux_external_product, &cmux_assignment_end);
  SEGMENT_DECLARE(cmux_decomposition, &cmux_external_product);
  SEGMENT_DECLARE(cmux_vector_matrix_product, &cmux_external_product);
  SEGMENT_DECLARE(cmux_fft_malloc, &cmux_vector_matrix_product);
  SEGMENT_DECLARE(cmux_fft_fft, &cmux_vector_matrix_product);
  SEGMENT_DECLARE(cmux_add_poly, &cmux_vector_matrix_product);
  SEGMENT_DECLARE(cmux_fourier_result, &cmux_external_product);
  SEGMENT_DECLARE(sample_extract, &root);
  SEGMENT_DECLARE(outro, &root);

  SEGMENT_START_ROOT(root);
  SEGMENT_START(intro);

  // The TLU is encoded and expanded, but it is not yet represented as a
  // ciphertext. We perform the trivial encryption of the tlu.
  uint32_t glwe_size = glwe_dimension + 1;
  uint64_t accumulator_size = polynomial_size * glwe_size;
  uint64_t *accumulator =
      (uint64_t *)malloc(accumulator_size * sizeof(uint64_t));
  uint64_t *accumulator_buf =
      (uint64_t *)malloc(accumulator_size * sizeof(uint64_t));
  uint64_t *tlu = tlu_aligned + tlu_offset;
  for (size_t i = 0; i < polynomial_size * glwe_dimension; i++) {
    accumulator[i] = 0;
  }
  for (size_t i = 0; i < polynomial_size; i++) {
    accumulator[polynomial_size * glwe_dimension + i] = tlu[i];
  }

  // Input ciphertext
  uint64_t *input_ct = ct0_aligned + ct0_offset;
  uint64_t input_size = input_lwe_dimension + 1;

  // Output ciphertext
  uint64_t *output_ct = out_aligned + out_offset;
  uint64_t output_size = glwe_dimension * polynomial_size + 1;

  // Bootstrap key
  uint32_t fourier_polynomial_size = polynomial_size / 2;

  // For the time being
  // Get fourrier bootstrap key

  // Get stack parameter
  size_t scratch_size;
  size_t scratch_align;
  concrete_cpu_bootstrap_lwe_ciphertext_u64_scratch(
      &scratch_size, &scratch_align, glwe_dimension, polynomial_size, fft);
  // Allocate scratch
  uint8_t *scratch = (uint8_t *)aligned_alloc(scratch_align, scratch_size);

  SEGMENT_END(intro);
  SEGMENT_START(blind_rotate);

  // ----------------------------------------------------------------------------------
  // BLIND ROTATE
  {
    // We begin by rotating the accumulator by multiplying the accumulator by
    // X^{-b}
    {
      // We get the body value
      uint64_t input_body = input_ct[input_size - 1];
      // We perform the modulus switch
      uint64_t monomial_degree = input_body;
      monomial_degree >>= 62 - ilog2(polynomial_size);
      monomial_degree += (uint64_t)1;
      monomial_degree >>= 1;
      // We copy the accumulator
      uint64_t *accumulator_ = accumulator_buf;
      memcpy(accumulator_, accumulator, accumulator_size * sizeof(uint64_t));
      // We perform the rotation of the lut polynomials
      uint64_t remaining_degree = monomial_degree % polynomial_size;
      uint64_t full_cycles_count = monomial_degree / polynomial_size;
      for (size_t i = 0; i < glwe_size; i++) {
        uint64_t *accumulator_poly = accumulator + i * polynomial_size;
        uint64_t *accumulator_poly_ = accumulator_ + i * polynomial_size;
        if (full_cycles_count % 2 == 0) {
          for (size_t j = 0; j < polynomial_size - remaining_degree; j++) {
            accumulator_poly[j] = accumulator_poly_[remaining_degree + j];
          }
          for (size_t j = 0; j < remaining_degree; j++) {
            accumulator_poly[polynomial_size - remaining_degree + j] =
                -accumulator_poly_[j];
          }
        } else {
          for (size_t j = 0; j < polynomial_size - remaining_degree; j++) {
            accumulator_poly[j] = -accumulator_poly_[remaining_degree + j];
          }
          for (size_t j = 0; j < remaining_degree; j++) {
            accumulator_poly[polynomial_size - remaining_degree + j] =
                accumulator_poly_[j];
          }
        }
      }
    }

    // We now execute the successive rotations of the accumulator by multiplying
    // by X^{a_i s_i}
    {
      uint64_t *ct0 = accumulator;
      uint64_t *ct1 = accumulator_buf;

      // We iterate through the mask elements
      {
        for (size_t i = 0; i < input_lwe_dimension; i++) {
          uint64_t lwe_mask_element = input_ct[i];
          const c64 *bootstrap_key_ggsw =
              bootstrap_key + i * decomposition_level_count *
                                  fourier_polynomial_size * glwe_size *
                                  glwe_size;

          if (lwe_mask_element != 0) {
            // We perform the modulus switching of the mask element.
            uint64_t monomial_degree = lwe_mask_element;
            monomial_degree >>= 62 - ilog2(polynomial_size);
            monomial_degree += (uint64_t)1;
            monomial_degree >>= 1;
            // --------------------------------------------------------------------------------
            // CMUX
            SEGMENT_START(cmux);
            {
              // We assign ct_1 <- ct_0 * X^{a_i} - ct_0 (the first step of the
              // cmux)
              SEGMENT_START(cmux_assignment_start);
              {
                for (size_t j = 0; j < glwe_size; j++) {
                  uint64_t *ct1_poly = ct1 + j * polynomial_size;
                  uint64_t *ct0_poly = ct0 + j * polynomial_size;

                  uint64_t remaining_degree = monomial_degree % polynomial_size;
                  uint64_t full_cycles_count =
                      monomial_degree / polynomial_size;
                  if (full_cycles_count % 2 == 0) {
                    for (size_t k = 0; k < remaining_degree; k++) {
                      ct1_poly[k] =
                          ((uint64_t)-ct0_poly[polynomial_size -
                                               remaining_degree + k]) -
                          (uint64_t)ct0_poly[k];
                    }
                    for (size_t k = 0; k < polynomial_size - remaining_degree;
                         k++) {
                      ct1_poly[remaining_degree + k] =
                          ct0_poly[k] - ct0_poly[remaining_degree + k];
                    }
                  } else {
                    for (size_t k = 0; k < remaining_degree; k++) {
                      ct1_poly[k] =
                          ct0_poly[polynomial_size - remaining_degree + k] -
                          ct0_poly[k];
                    }
                    for (size_t k = 0; k < polynomial_size - remaining_degree;
                         k++) {
                      ct1_poly[remaining_degree + k] =
                          ((uint64_t)-ct0_poly[k]) -
                          (uint64_t)ct0_poly[remaining_degree + k];
                    }
                  }
                }
              }
              SEGMENT_END(cmux_assignment_start);
              SEGMENT_START(cmux_assignment_end);
              // We assign ct_0 <- ct_0 + s_i ct_1 (the end of the cmux)
              {
                // We allocate a buffer to store the result
                c64 *output_fft_buffer = (c64 *)malloc(fourier_polynomial_size *
                                                       glwe_size * sizeof(c64));

                bool output_fft_buffer_initialized = false;

                // We compute ct0 closest representable under the decomposition.
                uint64_t *ct1_closest_representable =
                    (uint64_t *)malloc(accumulator_size * sizeof(uint64_t));
                for (size_t j = 0; j < accumulator_size; j++) {
                  uint64_t non_rep_bit_count =
                      64 - decomposition_level_count * decomposition_base_log;
                  uint64_t shift = non_rep_bit_count - 1;
                  uint64_t closest = ct1[j] >> shift;
                  closest += (uint64_t)1;
                  closest &= (uint64_t)-2;
                  closest <<= shift;
                  ct1_closest_representable[j] = closest;
                }

                //-----------------------------------------------------------------
                // EXTERNAL PRODUCT
                SEGMENT_START(cmux_external_product);
                // external_product_c(fft, bootstrap_key_ggsw,
                //                    ct1_closest_representable, scratch,
                //                    scratch_size, output_fft_buffer, ct0,
                //                    polynomial_size,
                //                    decomposition_level_count,
                //                    decomposition_base_log, glwe_dimension);
                {
                  // We prepare for the decomposition of the input glwe
                  uint64_t *glwe = ct1_closest_representable;
                  uint64_t dec_mod_b_mask =
                      ((uint64_t)1 << decomposition_base_log) - (uint64_t)1;
                  uint64_t *dec_state =
                      (uint64_t *)malloc(accumulator_size * sizeof(uint64_t));
                  uint64_t *glwe_decomp =
                      (uint64_t *)malloc(accumulator_size * sizeof(uint64_t));
                  for (size_t j = 0; j < accumulator_size; j++) {
                    dec_state[j] = glwe[j] >> (64 - decomposition_level_count *
                                                        decomposition_base_log);
                  }

                  // We iterate through the levels (in reverse order because
                  // that's how decomposition operates)
                  for (int level = decomposition_level_count - 1; level >= 0;
                       level--) {
                    SEGMENT_START(cmux_decomposition);
                    const c64 *ggsw_decomp_matrix =
                        bootstrap_key_ggsw +
                        level * fourier_polynomial_size * glwe_size * glwe_size;
                    for (size_t k = 0; k < accumulator_size; k++) {
                      uint64_t decomposed = dec_state[k] & dec_mod_b_mask;
                      dec_state[k] >>= decomposition_base_log;
                      uint64_t carry =
                          ((decomposed - (uint64_t)1) | dec_state[k]) &
                          decomposed;
                      carry >>= decomposition_base_log - 1;
                      dec_state[k] += carry;
                      decomposed -= carry << decomposition_base_log;
                      glwe_decomp[k] = decomposed;
                    }
                    SEGMENT_END(cmux_decomposition);

                    SEGMENT_START(cmux_vector_matrix_product);

                    // We perform the vector matrix product
                    for (size_t k = 0; k < glwe_size; k++) {
                      const c64 *ggsw_decomp_row =
                          ggsw_decomp_matrix +
                          k * fourier_polynomial_size * glwe_size;
                      uint64_t *glwe_decomp_poly =
                          glwe_decomp + k * polynomial_size;

                      SEGMENT_START(cmux_fft_malloc);
                      // We perform the fft
                      c64 *fourier_glwe_decomp_poly =
                          (c64 *)malloc(fourier_polynomial_size * sizeof(c64));

                      SEGMENT_END(cmux_fft_malloc);
                      SEGMENT_START(cmux_fft_fft);
                      concrete_cpu_fft(fourier_glwe_decomp_poly,
                                       glwe_decomp_poly, polynomial_size, fft,
                                       scratch, scratch_size);
                      SEGMENT_END(cmux_fft_fft);

                      SEGMENT_START(cmux_add_poly);

                      // We loop through the polynomials of the output and add
                      // the corresponding product of polynomials.
                      for (size_t m = 0; m < glwe_size; m++) {
                        const c64 *ggsw_decomp_poly =
                            ggsw_decomp_row + m * fourier_polynomial_size;
                        c64 *output_fft_buffer_poly =
                            output_fft_buffer + m * fourier_polynomial_size;

                        // We accumulate the product inside the output buffer.
                        if (!output_fft_buffer_initialized) {
                          for (size_t n = 0; n < fourier_polynomial_size; n++) {
                            output_fft_buffer_poly[n] =
                                c64_mul(ggsw_decomp_poly[n],
                                        fourier_glwe_decomp_poly[n]);
                          }
                        } else {
                          for (size_t n = 0; n < fourier_polynomial_size; n++) {
                            c64 mul = c64_mul(ggsw_decomp_poly[n],
                                              fourier_glwe_decomp_poly[n]);
                            output_fft_buffer_poly[n].c[0] += mul.c[0];
                            output_fft_buffer_poly[n].c[1] += mul.c[1];
                          }
                        }
                      }
                      SEGMENT_END(cmux_add_poly);

                      output_fft_buffer_initialized = true;
                      free(fourier_glwe_decomp_poly);
                    }

                    SEGMENT_END(cmux_vector_matrix_product);
                  }

                  SEGMENT_START(cmux_fourier_result);
                  // We retrieve the result from the fourier domain
                  for (size_t j = 0; j < glwe_size; j++) {
                    c64 *output_fft_buffer_poly =
                        output_fft_buffer + j * fourier_polynomial_size;
                    uint64_t *ct0_poly = ct0 + j * polynomial_size;
                    // We directly add the inverse fft to ct0
                    concrete_cpu_add_ifft(ct0_poly, output_fft_buffer_poly,
                                          polynomial_size, fft, scratch,
                                          scratch_size);
                  }
                  SEGMENT_END(cmux_fourier_result);

                  free(dec_state);
                  free(glwe_decomp);
                }
                SEGMENT_END(cmux_external_product);

                free(ct1_closest_representable);
                free(output_fft_buffer);
              }
              SEGMENT_END(cmux_assignment_end);
            }
            SEGMENT_END(cmux);
          }
        }
      }
    }
  }

  SEGMENT_END(blind_rotate);
  SEGMENT_START(sample_extract);

  // --------------------------------------------------------------------------------
  // SAMPLE EXTRACT
  {
    // We copy the mask and the first coefficient of the body of the accumulator
    // glwe to the output lwe.
    memcpy(output_ct, accumulator, output_size * sizeof(uint64_t));

    // We compute the number of elements that must be turned into their opposite
    uint64_t opposite_count = polynomial_size - 1;

    // We loop through the polynomials
    for (size_t i = 0; i < glwe_dimension; i++) {
      uint64_t *output_ct_mask_poly = output_ct + i * polynomial_size;
      // We reverse the polynomial
      for (size_t j = 0; j < polynomial_size / 2; j++) {
        uint64_t tmp = output_ct_mask_poly[j];
        output_ct_mask_poly[j] = output_ct_mask_poly[polynomial_size - 1 - j];
        output_ct_mask_poly[polynomial_size - 1 - j] = tmp;

        /* std::swap(output_ct_mask_poly[j], */
        /*           output_ct_mask_poly[polynomial_size - 1 - j]); */
      }
      // We negate the proper coefficients
      for (size_t j = 0; j < opposite_count; j++) {
        output_ct_mask_poly[j] = ((uint64_t)-output_ct_mask_poly[j]);
      }
      // Last, we have to rotate the array one position to the right
      uint64_t temp = output_ct_mask_poly[opposite_count];
      for (size_t j = opposite_count; j >= 1; j--) {
        output_ct_mask_poly[j] = output_ct_mask_poly[j - 1];
      }
      output_ct_mask_poly[0] = temp;
    }
  }

  SEGMENT_END(sample_extract);
  SEGMENT_START(outro);

  free(accumulator);
  free(accumulator_buf);
  free(scratch);

  SEGMENT_END(outro);
  SEGMENT_END_ROOT(root);
  SEGMENT_STATS(&root);
}

void memref_bootstrap_lwe_u64_c(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dimension, uint32_t polynomial_size,
    uint32_t decomposition_level_count, uint32_t decomposition_base_log,
    uint32_t glwe_dimension, uint32_t bsk_index, const c64 *bootstrap_key,
    const Fft *fft) {

  /* printf("out_size: %" PRIu64 "\n" */
  /*        "ct0_size: %" PRIu64 "\n" */
  /*        "tlu_size: %" PRIu64 "\n" */
  /*        "input_lwe_dimension: %" PRIu32 "\n" */
  /*        "polynomial_size: %" PRIu32 "\n" */
  /*        "decomposition_level_count: %" PRIu32 "\n" */
  /*        "decomposition_base_log: %" PRIu32 "\n" */
  /*        "glwe_dimension: %" PRIu32 "\n" */
  /*        "bsk_index: %" PRIu32 "\n", */

  /*        out_size, ct0_size, tlu_size, input_lwe_dimension, polynomial_size, */
  /*        decomposition_level_count, decomposition_base_log, glwe_dimension, */
  /*        bsk_index); */

  assert(ct0_size == input_lwe_dimension + 1);

  uint64_t(*foo)[100] = (uint64_t(*)[100])tlu_aligned;

  if (out_size == OPT_OUT_SIZE &&
      decomposition_level_count == OPT_DECOMPOSITION_LEVEL_COUNT &&
      decomposition_base_log == OPT_DECOMPOSITION_BASE_LOG &&
      glwe_dimension == OPT_GLWE_DIMENSION && tlu_size == OPT_TLU_SIZE &&
      input_lwe_dimension == OPT_INPUT_LWE_DIMENSION &&
      polynomial_size == OPT_POLYNOMIAL_SIZE &&

      out_stride == 1 && ct0_stride == 1 && tlu_stride == 1) {
    memref_bootstrap_lwe_u64_c_optimized(
        out_aligned + out_offset, ct0_aligned + ct0_offset,
        tlu_aligned + tlu_offset, (void *)bootstrap_key, fft);
  } else {
    memref_bootstrap_lwe_u64_c_generic(
        out_allocated, out_aligned, out_offset, out_size, out_stride,
        ct0_allocated, ct0_aligned, ct0_offset, ct0_size, ct0_stride,
        tlu_allocated, tlu_aligned, tlu_offset, tlu_size, tlu_stride,
        input_lwe_dimension, polynomial_size, decomposition_level_count,
        decomposition_base_log, glwe_dimension, bootstrap_key, fft);
  }
}
