//#define ROOT_PROFILING
//#define NO_PAPI_PROFILING

#include "bootstrap_optimized.h"
#include "bootstrap.h"
#include "c64_ops.h"
#include "segment.h"
#include <assert.h>
#include <fftw.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "twisties.h"
#include "unit_roots.h"

static inline size_t trailing_zeros(size_t i) { return __builtin_ctz(i); }

static inline size_t rev(size_t i) {
  static const uint8_t rev_bytes[256] = {
      0,  128, 64, 192, 32, 160, 96,  224, 16, 144, 80, 208, 48, 176, 112, 240,
      8,  136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
      4,  132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
      12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
      2,  130, 66, 194, 34, 162, 98,  226, 18, 146, 82, 210, 50, 178, 114, 242,
      10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
      6,  134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
      14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
      1,  129, 65, 193, 33, 161, 97,  225, 17, 145, 81, 209, 49, 177, 113, 241,
      9,  137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
      5,  133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
      13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
      3,  131, 67, 195, 35, 163, 99,  227, 19, 147, 83, 211, 51, 179, 115, 243,
      11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
      7,  135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
      15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255};

  size_t res = 0;

  for (size_t byte = 0; byte < sizeof(i) - 1; byte++) {
    res |= rev_bytes[i & 0xff];
    res <<= 8;
    i >>= 8;
  }

  return res;
}

static inline size_t bit_rev(size_t nbits, size_t i) {
  return rev(i) >> ((sizeof(i) * 8) - nbits);
}

void iterative_fft_dit(c64 out[OPT_FOURIER_POLYNOMIAL_SIZE],
                       const c64 in[OPT_FOURIER_POLYNOMIAL_SIZE]) {
  size_t log2n = trailing_zeros(OPT_FOURIER_POLYNOMIAL_SIZE);

  /* SEGMENT_DECLARE_ROOT(root); */
  /* SEGMENT_DECLARE(reverse, &root); */
  /* SEGMENT_DECLARE(fft, &root); */

  /* SEGMENT_START_ROOT(root); */
  /* SEGMENT_START(reverse); */

  for (size_t i = 0; i < OPT_FOURIER_POLYNOMIAL_SIZE; i++) {
    out[i] = in[bit_rev(log2n, i)];
  }

  /* SEGMENT_END(reverse); */
  /* SEGMENT_START(fft); */

  for (size_t s = 1; s <= log2n; s++) {
    size_t m = 1 << s;
    size_t twisty_base = 1 << (log2n - s);

    for (size_t k = 0; k < OPT_FOURIER_POLYNOMIAL_SIZE / m; k++) {
      for (size_t j = 0; j < m / 2; j++) {
        c64 twisty = twisties[twisty_base * j];
        c64 q = c64_mul(twisty, out[k * m + j + m / 2]);
        c64 p = out[k * m + j];
        out[k * m + j] = c64_add(p, q);
        out[k * m + j + m / 2] = c64_sub(p, q);
      }
    }
  }

  /* SEGMENT_END(fft); */
  /* SEGMENT_END_ROOT(root); */

  /* SEGMENT_STATS_TC(&root, 100); */
}

void iterative_ifft(c64 out[OPT_FOURIER_POLYNOMIAL_SIZE],
                    const c64 in[OPT_FOURIER_POLYNOMIAL_SIZE]) {
  size_t log2n = trailing_zeros(OPT_FOURIER_POLYNOMIAL_SIZE);

  /* SEGMENT_DECLARE_ROOT(root); */
  /* SEGMENT_DECLARE(reverse, &root); */
  /* SEGMENT_DECLARE(fft, &root); */

  /* SEGMENT_START_ROOT(root); */
  /* SEGMENT_START(reverse); */

  for (size_t i = 0; i < OPT_FOURIER_POLYNOMIAL_SIZE; i++) {
    out[i] = in[bit_rev(log2n, i)];
  }

  /* SEGMENT_END(reverse); */
  /* SEGMENT_START(fft); */

  for (size_t s = 1; s <= log2n; s++) {
    size_t m = 1 << s;
    size_t twisty_base = 1 << (log2n - s);

    for (size_t k = 0; k < OPT_FOURIER_POLYNOMIAL_SIZE / m; k++) {
      for (size_t j = 0; j < m / 2; j++) {
        c64 twisty = c64_inv(twisties[twisty_base * j]);
        c64 q = c64_mul(twisty, out[k * m + j + m / 2]);
        c64 p = out[k * m + j];
        out[k * m + j] = c64_add(p, q);
        out[k * m + j + m / 2] = c64_sub(p, q);
      }
    }
  }

  for (size_t i = 0; i < OPT_FOURIER_POLYNOMIAL_SIZE; i++) {
    out[i] = c64_rdiv(out[i], OPT_FOURIER_POLYNOMIAL_SIZE);
  }

  /* SEGMENT_END(fft); */
  /* SEGMENT_END_ROOT(root); */

  /* SEGMENT_STATS_TC(&root, 100); */
}

static inline size_t bit_rev_twice_inv(size_t nbits, size_t base_nbits,
                                       size_t i) {
  size_t bottom_mask = ((size_t)1 << base_nbits) - 1;
  size_t bottom_bits = bit_rev(base_nbits, i);
  size_t i_rev = (i & ~bottom_mask) | bottom_bits;

  // 0000.1111.1111.1110 to 0000.0011.1111.1111
  return bit_rev(nbits, i_rev);
}

void print_binary_lsb(size_t i, size_t nbits) {
  assert(nbits % 4 == 0);

  static const char *bits[] = {"0000", "0001", "0010", "0011", "0100", "0101",
                               "0110", "0111", "1000", "1001", "1010", "1011",
                               "1100", "1101", "1110", "1111"};

  for (size_t k = (sizeof(i) * 8 - nbits) + 4; k <= sizeof(i) * 8; k += 4) {
    printf("%s%s", bits[(i >> (sizeof(i) * 8 - k)) & 0xf],
           k == sizeof(i) * 8 ? "" : ".");
  }
}

void print_binary(size_t i) { print_binary_lsb(i, sizeof(i) * 8); }

static inline void one_concrete_cpu_fft(c64 *out, const uint64_t *in, size_t n,
                                        void *scratch, size_t scratch_size,
                                        const void *thefft) {
  SEGMENT_DECLARE_ROOT(root);
  SEGMENT_DECLARE(fft, &root);

  SEGMENT_START_ROOT(root);
  SEGMENT_START(fft);

  concrete_cpu_fft(out, in, OPT_POLYNOMIAL_SIZE, thefft, scratch, scratch_size);

  SEGMENT_END(fft);
  SEGMENT_END_ROOT(root);

  SEGMENT_STATS(&root);
}

static inline void c_fft(c64 *out, const uint64_t *in) {
  /* SEGMENT_DECLARE_ROOT(root); */
  /* SEGMENT_DECLARE(intro, &root); */
  /* SEGMENT_DECLARE(preconditioning, &root); */
  /* SEGMENT_DECLARE(fft, &root); */
  /* SEGMENT_DECLARE(permutation, &root); */
  /* SEGMENT_DECLARE(outro, &root); */

  /* SEGMENT_START_ROOT(root); */
  /* SEGMENT_START(intro); */
  const int64_t *in_signed = (const int64_t *)in;

  c64 tmp[OPT_FOURIER_POLYNOMIAL_SIZE];

  /* SEGMENT_END(intro); */
  /* SEGMENT_START(preconditioning); */

  /* from fn convert_forward_integer_scalar<Scalar: UnsignedTorus> */
  for (size_t i = 0; i < OPT_FOURIER_POLYNOMIAL_SIZE; i++) {
    tmp[i] = c64_mul(
        c64_build(in_signed[i], in_signed[i + OPT_FOURIER_POLYNOMIAL_SIZE]),
        unit_roots[i]);
  }

  /* SEGMENT_END(preconditioning); */

  /* SEGMENT_START(fft); */
  iterative_fft_dit(out, tmp);
  /* SEGMENT_END(fft); */

  /* SEGMENT_START(permutation); */

  size_t nbits = trailing_zeros(OPT_FOURIER_POLYNOMIAL_SIZE);
  size_t base_n = 512;
  size_t base_nbits = trailing_zeros(base_n);

  for (size_t i = 0; i < OPT_FOURIER_POLYNOMIAL_SIZE; i++) {
    size_t idx = bit_rev_twice_inv(nbits, base_nbits, i);
    tmp[i] = out[idx];
  }

  /* SEGMENT_END(permutation); */
  /* SEGMENT_START(outro); */

  memcpy(out, tmp, sizeof(tmp));

  /* SEGMENT_END(outro); */
  /* SEGMENT_END_ROOT(root); */

  // SEGMENT_STATS_TC(&root, 100);
}

static uint32_t ilog2(uint32_t val) { return 31 - __builtin_clz(val); }

void memref_bootstrap_lwe_u64_c_optimized(
    uint64_t output_ct[OPT_OUT_SIZE], uint64_t input_ct[OPT_INPUT_SIZE],
    uint64_t tlu[OPT_TLU_SIZE],

    // Size: 894 * 1 * 2 * 2 * 4096 * 16 ~= 224 MiB
    const c64 bootstrap_key[OPT_INPUT_LWE_DIMENSION]
                           [OPT_DECOMPOSITION_LEVEL_COUNT][OPT_GLWE_SIZE]
                           [OPT_GLWE_SIZE][OPT_FOURIER_POLYNOMIAL_SIZE],
    const Fft *fft) {
  puts("C bootstrap optimized");

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
  SEGMENT_DECLARE(cmux_fft_fft, &cmux_vector_matrix_product);
  SEGMENT_DECLARE(cmux_add_poly, &cmux_vector_matrix_product);
  SEGMENT_DECLARE(cmux_fourier_result, &cmux_external_product);
  SEGMENT_DECLARE(sample_extract, &root);
  SEGMENT_DECLARE(outro, &root);

  SEGMENT_START_ROOT(root);
  SEGMENT_START(intro);

  // The TLU is encoded and expanded, but it is not yet represented as a
  // ciphertext. We perform the trivial encryption of the tlu.
  static uint64_t accumulatorX[OPT_GLWE_SIZE][OPT_POLYNOMIAL_SIZE];
  static uint64_t accumulator_bufX[OPT_GLWE_SIZE][OPT_POLYNOMIAL_SIZE];

  memset(accumulatorX, 0, sizeof(accumulatorX));

  for (size_t i = 0; i < OPT_POLYNOMIAL_SIZE; i++) {
    accumulatorX[OPT_GLWE_SIZE - 1][i] = tlu[i];
  }

  // Input ciphertext

  // Output ciphertext
  uint64_t output_size = OPT_GLWE_DIMENSION * OPT_POLYNOMIAL_SIZE + 1;

  // Bootstrap key
  // For the time being
  // Get fourrier bootstrap key

  // Get stack parameter
  // 868 KB
  size_t scratch_size;
  size_t scratch_align;
  concrete_cpu_bootstrap_lwe_ciphertext_u64_scratch(
      &scratch_size, &scratch_align, OPT_GLWE_DIMENSION, OPT_POLYNOMIAL_SIZE,
      fft);
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
      uint64_t input_body = input_ct[OPT_INPUT_SIZE - 1];
      // We perform the modulus switch
      uint64_t monomial_degree = input_body;
      monomial_degree >>= 62 - ilog2(OPT_POLYNOMIAL_SIZE);
      monomial_degree += (uint64_t)1;
      monomial_degree >>= 1;
      // We copy the accumulator

      memcpy(accumulator_bufX, accumulatorX, sizeof(accumulatorX));

      // We perform the rotation of the lut polynomials
      uint64_t remaining_degree = monomial_degree % OPT_POLYNOMIAL_SIZE;
      uint64_t full_cycles_count = monomial_degree / OPT_POLYNOMIAL_SIZE;

      for (size_t i = 0; i < OPT_GLWE_SIZE; i++) {
        if (full_cycles_count % 2 == 0) {
          for (size_t j = 0; j < OPT_POLYNOMIAL_SIZE - remaining_degree; j++) {
            accumulatorX[i][j] = accumulator_bufX[i][remaining_degree + j];
          }
          for (size_t j = 0; j < remaining_degree; j++) {
            accumulatorX[i][OPT_POLYNOMIAL_SIZE - remaining_degree + j] =
                -accumulator_bufX[i][j];
          }
        } else {
          for (size_t j = 0; j < OPT_POLYNOMIAL_SIZE - remaining_degree; j++) {
            accumulatorX[i][j] = -accumulator_bufX[i][remaining_degree + j];
          }
          for (size_t j = 0; j < remaining_degree; j++) {
            accumulatorX[i][OPT_POLYNOMIAL_SIZE - remaining_degree + j] =
                accumulator_bufX[i][j];
          }
        }
      }
    }

    // We now execute the successive rotations of the accumulator by
    // multiplying by X^{a_i s_i}
    {
      // We iterate through the mask elements
      {
        for (size_t i = 0; i < OPT_INPUT_LWE_DIMENSION; i++) {
          uint64_t lwe_mask_element = input_ct[i];

          if (lwe_mask_element != 0) {
            // We perform the modulus switching of the mask element.
            uint64_t monomial_degree = lwe_mask_element;
            monomial_degree >>= 62 - ilog2(OPT_POLYNOMIAL_SIZE);
            monomial_degree += (uint64_t)1;
            monomial_degree >>= 1;
            // --------------------------------------------------------------------------------
            // CMUX
            SEGMENT_START(cmux);
            {
              // We assign ct_1 <- ct_0 * X^{a_i} - ct_0 (the first step of
              // the cmux)
              SEGMENT_START(cmux_assignment_start);
              {
                for (size_t j = 0; j < OPT_GLWE_SIZE; j++) {
                  uint64_t remaining_degree =
                      monomial_degree % OPT_POLYNOMIAL_SIZE;
                  uint64_t full_cycles_count =
                      monomial_degree / OPT_POLYNOMIAL_SIZE;
                  if (full_cycles_count % 2 == 0) {
                    for (size_t k = 0; k < remaining_degree; k++) {
                      accumulator_bufX[j][k] =
                          ((uint64_t)-accumulatorX[j][OPT_POLYNOMIAL_SIZE -
                                                      remaining_degree + k]) -
                          (uint64_t)accumulatorX[j][k];
                    }
                    for (size_t k = 0;
                         k < OPT_POLYNOMIAL_SIZE - remaining_degree; k++) {
                      accumulator_bufX[j][remaining_degree + k] =
                          accumulatorX[j][k] -
                          accumulatorX[j][remaining_degree + k];
                    }
                  } else {
                    for (size_t k = 0; k < remaining_degree; k++) {
                      accumulator_bufX[j][k] =
                          accumulatorX[j][OPT_POLYNOMIAL_SIZE -
                                          remaining_degree + k] -
                          accumulatorX[j][k];
                    }
                    for (size_t k = 0;
                         k < OPT_POLYNOMIAL_SIZE - remaining_degree; k++) {
                      accumulator_bufX[j][remaining_degree + k] =
                          ((uint64_t)-accumulatorX[j][k]) -
                          (uint64_t)accumulatorX[j][remaining_degree + k];
                    }
                  }
                }
              }
              SEGMENT_END(cmux_assignment_start);
              SEGMENT_START(cmux_assignment_end);
              // We assign ct_0 <- ct_0 + s_i ct_1 (the end of the cmux)
              {
                // We allocate a buffer to store the result
                // Size: 2 * 4096 * 16 = 128 KiB
                static c64 output_fft_buffer[OPT_GLWE_SIZE]
                                            [OPT_FOURIER_POLYNOMIAL_SIZE];

                memset(output_fft_buffer, 0,
                       OPT_GLWE_SIZE * OPT_FOURIER_POLYNOMIAL_SIZE *
                           sizeof(c64));

                // We compute ct0 closest representable under the
                // decomposition.

                static const uint64_t non_rep_bit_count =
                    64 -
                    OPT_DECOMPOSITION_LEVEL_COUNT * OPT_DECOMPOSITION_BASE_LOG;
                static const uint64_t shift = non_rep_bit_count - 1;
                static uint64_t ct1_closest_representable[OPT_GLWE_SIZE]
                                                         [OPT_POLYNOMIAL_SIZE];

                for (size_t i = 0; i < OPT_GLWE_SIZE; i++) {
                  for (size_t j = 0; j < OPT_POLYNOMIAL_SIZE; j++) {
                    uint64_t closest = accumulator_bufX[i][j] >> shift;
                    closest += (uint64_t)1;
                    closest &= (uint64_t)-2;
                    closest <<= shift;
                    ct1_closest_representable[i][j] = closest;
                  }
                }

                //-----------------------------------------------------------------
                // EXTERNAL PRODUCT
                SEGMENT_START(cmux_external_product);
                {
                  // We prepare for the decomposition of the input glwe
                  /* uint64_t *glwe = ct1_closest_representable; */
                  uint64_t dec_mod_b_mask =
                      ((uint64_t)1 << OPT_DECOMPOSITION_BASE_LOG) - (uint64_t)1;

                  // 128 KiB
                  static uint64_t dec_state[OPT_GLWE_SIZE][OPT_POLYNOMIAL_SIZE];
                  // 128 KiB
                  static uint64_t glwe_decomp[OPT_GLWE_SIZE]
                                             [OPT_POLYNOMIAL_SIZE];

                  for (size_t i = 0; i < OPT_GLWE_SIZE; i++) {
                    for (size_t j = 0; j < OPT_POLYNOMIAL_SIZE; j++) {
                      dec_state[i][j] = ct1_closest_representable[i][j] >>
                                        (64 - OPT_DECOMPOSITION_LEVEL_COUNT *
                                                  OPT_DECOMPOSITION_BASE_LOG);
                    }
                  }

                  // We iterate through the levels (in reverse order because
                  // that's how decomposition operates)
                  for (int level = OPT_DECOMPOSITION_LEVEL_COUNT - 1;
                       level >= 0; level--) {
                    SEGMENT_START(cmux_decomposition);

                    // 16384 round-trips, WSS: 256 KiB
                    //
                    // cmux_decomposition:
                    //  13.90%R [11.14%T 12955549ns]
                    //  CPI: 0.34602
                    //  LDpi: 11.18%
                    //  L1pl: 25.08%
                    //  L2pl:  3.29%
                    //  L3pl:  0.01%
                    //  BRMR:  0.02%
                    //
                    // Embarrassingly parallel
                    for (size_t i = 0; i < OPT_GLWE_SIZE; i++) {
                      for (size_t k = 0; k < OPT_POLYNOMIAL_SIZE; k++) {
                        uint64_t decomposed = dec_state[i][k] & dec_mod_b_mask;
                        dec_state[i][k] >>= OPT_DECOMPOSITION_BASE_LOG;
                        uint64_t carry =
                            ((decomposed - (uint64_t)1) | dec_state[i][k]) &
                            decomposed;
                        carry >>= OPT_DECOMPOSITION_BASE_LOG - 1;
                        dec_state[i][k] += carry;
                        decomposed -= carry << OPT_DECOMPOSITION_BASE_LOG;
                        glwe_decomp[i][k] = decomposed;
                      }
                    }

                    SEGMENT_END(cmux_decomposition);

                    SEGMENT_START(cmux_vector_matrix_product);

                    // We perform the vector matrix product
                    for (size_t k = 0; k < OPT_GLWE_SIZE; k++) {
                      // We perform the fft
                      static c64
                          fourier_glwe_decomp_poly[OPT_FOURIER_POLYNOMIAL_SIZE]
                          __attribute__((aligned));

                      // cmux_fft_fft: 34.25%R [14.82%T 17238673ns]
                      //  CPI: 0.56120,
                      //  LDpi: 15.88%,
                      //  L1pl: 69.45%,
                      //  L2pl: 1.82%,
                      //  L3pl: 0.02%,
                      //  BRMR: 0.21%
                      SEGMENT_START(cmux_fft_fft);

                      // // WSS > 4096*16 + 128 KiB + 868 KB ~= 1 MiB
                      /* concrete_cpu_fft(fourier_glwe_decomp_poly /\* output
                       * *\/, */
                      /*                  &glwe_decomp[k][0] /\* input */
                      /*                                      *\/ */
                      /*                  , */
                      /*                  OPT_POLYNOMIAL_SIZE, fft, scratch, */
                      /*                  scratch_size); */

                      /* one_concrete_cpu_fft( */
                      /*     fourier_glwe_decomp_poly, &glwe_decomp[k][0], */
                      /*     OPT_POLYNOMIAL_SIZE, scratch, scratch_size, fft);
                       */

                      c_fft(fourier_glwe_decomp_poly, &glwe_decomp[k][0]);

                      SEGMENT_END(cmux_fft_fft);

                      SEGMENT_START(cmux_add_poly);

                      // We loop through the polynomials of the output and add
                      // the corresponding product of polynomials.
                      //
                      // 2*8192 round-trips, WSS: 224 MiB
                      //
                      // cmux_add_poly: 45.61%R [19.74%T 22960752ns]
                      //  CPI: 0.32730
                      //  LDpi: 23.53%
                      //  L1pl: 18.82%
                      //  L2pl: 6.08%
                      //  L3pl: 5.55%
                      //  BRMR: 0.04%
                      for (size_t m = 0; m < OPT_GLWE_SIZE; m++) {
                        for (size_t n = 0; n < OPT_FOURIER_POLYNOMIAL_SIZE;
                             n++) {
                          c64 mul = c64_mul(bootstrap_key[i][level][k][m][n],
                                            fourier_glwe_decomp_poly[n]);
                          output_fft_buffer[m][n].c[0] += mul.c[0];
                          output_fft_buffer[m][n].c[1] += mul.c[1];
                        }
                      }
                      SEGMENT_END(cmux_add_poly);
                    }

                    SEGMENT_END(cmux_vector_matrix_product);
                  }

                  SEGMENT_START(cmux_fourier_result);
                  // We retrieve the result from the fourier domain
                  //
                  // WSS > 8192*8 + 4096 * 8 + 868 KB ~= 1MiB
                  //
                  // 2 Roundtrips
                  //
                  // cmux_fourier_result: 19.36%R [15.54%T 23332022ns]
                  //  CPI: 0.57565
                  //  LDpi: 13.90%
                  //  L1pl: 63.49%
                  //  L2pl:  8.47%
                  //  L3pl:  0.03%
                  //  BRMR:  0.20%

                  for (size_t j = 0; j < OPT_GLWE_SIZE; j++) {
                    // We directly add the inverse fft to ct0

                    concrete_cpu_add_ifft(
                        &accumulatorX[j][0], &output_fft_buffer[j][0],
                        OPT_POLYNOMIAL_SIZE, fft, scratch, scratch_size);

                    /* static c64 tmp[OPT_FOURIER_POLYNOMIAL_SIZE]; */

                    /* iterative_ifft(&tmp[0], &output_fft_buffer[j][0]); */

                    /* accumulatorX[j][i] = (uint64_t)((int64_t)tmp[i].c[0]); */
                    /* accumulatorX[j][i + OPT_FOURIER_POLYNOMIAL_SIZE] = */
                    /*     (uint64_t)((int64_t)tmp[i].c[1]); */
                  }
                  SEGMENT_END(cmux_fourier_result);
                }
                SEGMENT_END(cmux_external_product);
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
    // We copy the mask and the first coefficient of the body of the
    // accumulator glwe to the output lwe.
    memcpy(output_ct, accumulatorX, output_size * sizeof(uint64_t));

    // We compute the number of elements that must be turned into their
    // opposite
    uint64_t opposite_count = OPT_POLYNOMIAL_SIZE - 1;

    // We loop through the polynomials
    for (size_t i = 0; i < OPT_GLWE_DIMENSION; i++) {
      uint64_t *output_ct_mask_poly = output_ct + i * OPT_POLYNOMIAL_SIZE;
      // We reverse the polynomial
      for (size_t j = 0; j < OPT_POLYNOMIAL_SIZE / 2; j++) {
        uint64_t tmp = output_ct_mask_poly[j];
        output_ct_mask_poly[j] =
            output_ct_mask_poly[OPT_POLYNOMIAL_SIZE - 1 - j];
        output_ct_mask_poly[OPT_POLYNOMIAL_SIZE - 1 - j] = tmp;
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

  free(scratch);

  SEGMENT_END(outro);
  SEGMENT_END_ROOT(root);
  SEGMENT_STATS(&root);
}
