#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

void keyswitch_lwe_generic(uint64_t *output_ct, uint64_t out_size,
                           uint64_t *input_ct, uint64_t ct0_size,
                           uint32_t decomposition_level_count,
                           uint32_t decomposition_base_log,
                           uint32_t input_dimension, uint32_t output_dimension,
                           const uint64_t *ksk) {
  puts("keyswitch_lwe_generic");

  // The input ciphertext is a vector of u64 of size input_dimension + 1. It has
  // the following memory repr: [a1, a2, ..., an, b]          with ai the mask
  // elements, and b the body.
  uint64_t input_size = input_dimension + 1;

  // The output ciphertext is a vector of u64 of size output_dimension + 1. It
  // has the same kind of as the input ct.
  uint64_t output_size = output_dimension + 1;

  // We retrieve the ksk from the context. It is a three dimensional contiguous
  // tensor with the following repr: [I1, I2, ..., Iv]             with v the
  // lwe dimension of the input key.
  //
  // Each Ii contains the encryptions of the decomposition of one bit of the
  // input key. It has the following repr: [D1, D2, ..., Dl]             with l
  // the number of decomposition levels used for the ksk.
  //
  // Each Di is the encryption under the output key, of one level of the
  // decomposition of the key bit. It has the same repr as athe : [D1, D2, ...,
  // Dl]             with l the number of decomposition levels used for the ksk.
  //  const uint64_t *ksk = context->keyswitch_key_buffer(ksk_index);

  // We compute the closest representable with the decomposition
  uint64_t non_rep_bit_count =
      64 - decomposition_level_count * decomposition_base_log;
  uint64_t shift = non_rep_bit_count - 1;
  uint64_t dec_mod_b_mask =
      ((uint64_t)1 << decomposition_base_log) - (uint64_t)1;

  // We begin by zeroing out the output ct
  for (size_t i = 0; i < output_size; i++) {
    output_ct[i] = 0;
  }

  // We copy the body of the input ct to the body of the output ct
  output_ct[output_size - 1] = input_ct[input_size - 1];

  // We loop through the mask elements of the input ct and the corresponding key
  // bits in the ksk.
  for (size_t i = 0; i < input_dimension; i++) {
    // We retrieve the block for the ith input key bit.
    const uint64_t *ksk_block =
        ksk + i * decomposition_level_count * output_size;
    //    const uint64_t *ksk_block3 = (const uint64_t *)&ksk3d[i][0][0];

    // We retrieve the mask element
    uint64_t mask_elm = input_ct[i];
    uint64_t closest = mask_elm >> shift;
    closest += (uint64_t)1;
    closest &= (uint64_t)-2;
    closest <<= shift;

    // We initialize the decomposition
    uint64_t dec_state = closest >> non_rep_bit_count;

    // We loop through the levels of the decomposition
    for (size_t j = 0; j < decomposition_level_count; j++) {
      // We retrieve the encryption of the jth decomposition of ith bit of the
      // input key.
      const uint64_t *ksk_ct = ksk_block + j * output_size;

      // We get the decomposed iterate
      uint64_t decomposed = dec_state & dec_mod_b_mask;
      dec_state >>= decomposition_base_log;
      uint64_t carry = ((decomposed - (uint64_t)1) | dec_state) & decomposed;
      carry >>= decomposition_base_log - 1;
      dec_state += carry;
      decomposed -= carry << decomposition_base_log;

      // We accumulate in the output ct
      for (size_t k = 0; k < output_size; k++) {
        output_ct[k] -= ksk_ct[k] * decomposed;
      }
    }
  }
}
void keyswitch_lwe_optimized(uint64_t *output_ct, uint64_t out_size,
                             uint64_t *input_ct, uint64_t ct0_size,
                             uint32_t decomposition_level_count,
                             uint32_t decomposition_base_log,
                             uint32_t input_dimension,
                             uint32_t output_dimension,
                             const uint64_t ksk3d[1536][3][709]) {
  puts("keyswitch_lwe_optimized");

  uint64_t input_size = 1536 + 1;

  uint64_t non_rep_bit_count = 64 - 3 * decomposition_base_log;
  uint64_t shift = non_rep_bit_count - 1;
  uint64_t dec_mod_b_mask =
      ((uint64_t)1 << decomposition_base_log) - (uint64_t)1;

  // We begin by zeroing out the output ct
  for (size_t i = 0; i < 709; i++) {
    output_ct[i] = 0;
  }

  // We copy the body of the input ct to the body of the output ct
  output_ct[709 - 1] = input_ct[input_size - 1];

  size_t i, j, k;
  uint64_t closest;

  //#pragma scop
  for (i = 0; i < 1536; i++) {
    // We retrieve the mask element
    closest = input_ct[i] >> shift;
    closest += (uint64_t)1;
    closest &= (uint64_t)-2;
    closest = closest << shift;

    // We initialize the decomposition
    uint64_t dec_state = closest >> non_rep_bit_count;

    // We loop through the levels of the decomposition
    for (j = 0; j < 3; j++) {
      // We get the decomposed iterate
      uint64_t decomposed = dec_state & dec_mod_b_mask;
      dec_state = dec_state >> decomposition_base_log;
      uint64_t carry = ((decomposed - (uint64_t)1) | dec_state) & decomposed;
      carry = carry >> (decomposition_base_log - 1);
      dec_state += carry;
      decomposed -= carry << decomposition_base_log;

      // We accumulate in the output ct
      for (k = 0; k < 709; k++) {
        output_ct[k] -= ksk3d[i][j][k] * decomposed;
      }
    }
  }
  //#pragma endscop
}

void keyswitch_lwe_c(uint64_t *out_allocated, uint64_t *out_aligned,
                     uint64_t out_offset, uint64_t out_size,
                     uint64_t out_stride, uint64_t *ct0_allocated,
                     uint64_t *ct0_aligned, uint64_t ct0_offset,
                     uint64_t ct0_size, uint64_t ct0_stride,
                     uint32_t decomposition_level_count,
                     uint32_t decomposition_base_log, uint32_t input_dimension,
                     uint32_t output_dimension, const uint64_t *ksk) {
  assert(out_stride == 1 && ct0_stride == 1);
  uint64_t *input_ct = ct0_aligned + ct0_offset;
  uint64_t *output_ct = out_aligned + out_offset;

  uint64_t output_size = output_dimension + 1;

  if (input_dimension == 1536 && decomposition_level_count == 3 &&
      output_size == 709) {
    keyswitch_lwe_optimized(output_ct, out_size, input_ct, ct0_size,
                            decomposition_level_count, decomposition_base_log,
                            input_dimension, output_dimension, (void *)ksk);
  } else {
    keyswitch_lwe_generic(output_ct, out_size, input_ct, ct0_size,
                          decomposition_level_count, decomposition_base_log,
                          input_dimension, output_dimension, ksk);
  }
}
