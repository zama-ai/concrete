#ifndef CUDA_MULT_H
#define CUDA_MULT_H

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#include "bootstrap.h"
#include "bootstrap_amortized.cuh"
#include "bootstrap_low_latency.cuh"
#include "device.h"
#include "keyswitch.cuh"
#include "linear_algebra.h"
#include "bootstrap_multibit.cuh"
#include "bootstrap_multibit.h"
#include "utils/kernel_dimensions.cuh"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
__global__ void
cleartext_multiplication(T *output, T *lwe_input, T *cleartext_input,
                         uint32_t input_lwe_dimension, uint32_t num_entries) {

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  if (index < num_entries) {
    int cleartext_index = index / (input_lwe_dimension + 1);
    // Here we take advantage of the wrapping behaviour of uint
    output[index] = lwe_input[index] * cleartext_input[cleartext_index];
  }
}

template <typename T>
__host__ void
host_cleartext_multiplication(void *v_stream, uint32_t gpu_index, T *output,
                              T *lwe_input, T *cleartext_input,
                              uint32_t input_lwe_dimension,
                              uint32_t input_lwe_ciphertext_count) {

  cudaSetDevice(gpu_index);
  // lwe_size includes the presence of the body
  // whereas lwe_dimension is the number of elements in the mask
  int lwe_size = input_lwe_dimension + 1;
  // Create a 1-dimensional grid of threads
  int num_blocks = 0, num_threads = 0;
  int num_entries = input_lwe_ciphertext_count * lwe_size;
  getNumBlocksAndThreads(num_entries, 512, num_blocks, num_threads);
  dim3 grid(num_blocks, 1, 1);
  dim3 thds(num_threads, 1, 1);

  auto stream = static_cast<cudaStream_t *>(v_stream);
  cleartext_multiplication<<<grid, thds, 0, *stream>>>(
      output, lwe_input, cleartext_input, input_lwe_dimension, num_entries);
  check_cuda_error(cudaGetLastError());
}

template <typename Torus, class params>
__global__ void fill(Torus *array, Torus value, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    array[idx] = value;
  }
}

int gcd(int a, int b) {
  if (b == 0)
    return a;
  else
    return gcd(b, a % b);
}


template <typename Torus, class params> void rotate_left(Torus *buffer, int p) {
  p = p % params::degree;
  int step = gcd(p, params::degree);
  for (int i = 0; i < step; i++) {
    int tmp = buffer[i];
    int j = i;

    while (1) {
      int k = j + p;
      if (k >= params::degree)
        k = k - params::degree;

      if (k == i)
        break;

      buffer[j] = buffer[k];
      j = k;
    }
    buffer[j] = tmp;
  }
}

template <typename Torus, class params>
void generate_lsb_msb_accumulators(
    void *v_stream, uint32_t gpu_index, Torus *lsb, Torus *msb,
    Torus *message_acc, Torus *carry_acc, Torus *tvi, Torus *tvi_message,
    Torus *tvi_carry, uint32_t glwe_dimension, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t lsb_count, uint32_t msb_count) {
  auto stream = static_cast<cudaStream_t *>(v_stream);

  uint32_t acc_size = params::degree * (glwe_dimension + 1) * sizeof(Torus);
  uint32_t tvi_size = (lsb_count + msb_count) * sizeof(Torus);
  Torus *h_lsb = (Torus *)malloc(acc_size);
  Torus *h_msb = (Torus *)malloc(acc_size);
  Torus *h_message = (Torus *)malloc(acc_size);
  Torus *h_carry = (Torus *)malloc(acc_size);

  Torus *h_tvi = (Torus *)malloc(tvi_size);
  Torus *h_tvi_message = (Torus *)malloc(tvi_size);
  Torus *h_tvi_carry = (Torus *)malloc(tvi_size);

  memset(h_lsb, 0, acc_size / 2);
  memset(h_msb, 0, acc_size / 2);
  memset(h_message, 0, acc_size / 2);
  memset(h_carry, 0, acc_size / 2);

  for (int i = 0; i < lsb_count + msb_count; i++) {
    h_tvi[i] = (i >= lsb_count);
    h_tvi_message[i] = 0;
    h_tvi_carry[i] = 0;
  }

  uint32_t modulus_sup = message_modulus * carry_modulus;
  uint32_t box_size = params::degree / modulus_sup;
  Torus delta = (1ul << 63) / modulus_sup;

  auto body_lsb = &h_lsb[params::degree];
  auto body_msb = &h_msb[params::degree];
  auto body_message = &h_message[params::degree];
  auto body_carry = &h_carry[params::degree];
  for (int i = 0; i < modulus_sup; i++) {
    int index = i * box_size;
    for (int j = index; j < index + box_size; j++) {
      Torus val_lsb = delta * ((i / message_modulus) * (i % message_modulus) %
                               message_modulus);
      Torus val_msb = delta * ((i / message_modulus) * (i % message_modulus) /
                               message_modulus);
      Torus val_message = delta * (i % message_modulus);
      Torus val_carry = delta * (i / message_modulus);
      body_lsb[j] = val_lsb;
      body_msb[j] = val_msb;
      body_message[j] = val_message;
      body_carry[j] = val_carry;
    }
  }

  rotate_left<Torus, params>(body_lsb, box_size / 2);
  rotate_left<Torus, params>(body_msb, box_size / 2);
  rotate_left<Torus, params>(body_message, box_size / 2);
  rotate_left<Torus, params>(body_carry, box_size / 2);

  cuda_memcpy_async_to_gpu(lsb, h_lsb, acc_size, stream, gpu_index);
  cuda_memcpy_async_to_gpu(msb, h_msb, acc_size, stream, gpu_index);
  cuda_memcpy_async_to_gpu(message_acc, h_message, acc_size, stream, gpu_index);
  cuda_memcpy_async_to_gpu(carry_acc, h_carry, acc_size, stream, gpu_index);

  cuda_memcpy_async_to_gpu(tvi, h_tvi, tvi_size, stream, gpu_index);
  cuda_memcpy_async_to_gpu(tvi_message, h_tvi_message, tvi_size, stream,
                           gpu_index);
  cuda_memcpy_async_to_gpu(tvi_carry, h_tvi_carry, tvi_size, stream, gpu_index);

  free(h_lsb);
  free(h_msb);
  free(h_message);
  free(h_carry);
  free(h_tvi);
  free(h_tvi_message);
  free(h_tvi_carry);
}

template <typename Torus, class params>
__global__ void tree_add_optimized(Torus *result_blocks, Torus *input_blocks,
                                   uint32_t glwe_dimension,
                                   uint32_t num_blocks) {
  size_t radix_size = num_blocks * (params::degree + 1);
  size_t big_lwe_id = blockIdx.x;
  size_t radix_id = big_lwe_id / num_blocks;
  size_t block_id = big_lwe_id % num_blocks;

  auto cur_res_radix = &result_blocks[radix_id * radix_size];
  auto cur_left_radix = &input_blocks[2 * radix_id * radix_size];
  auto cur_right_radix = &input_blocks[(2 * radix_id + 1) * radix_size];

  Torus *cur_res_block = &cur_res_radix[block_id * (params::degree + 1)];
  Torus *cur_left_block = &cur_left_radix[block_id * (params::degree + 1)];
  Torus *cur_right_block = &cur_right_radix[block_id * (params::degree + 1)];

  size_t tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    cur_res_block[tid] = cur_left_block[tid] + cur_right_block[tid];
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    cur_res_block[params::degree] =
        cur_left_block[params::degree] + cur_right_block[params::degree];
  }
}

template <typename Torus, class params>
__global__ void tree_add(Torus *result_blocks, Torus *input_blocks,
                         uint32_t glwe_dimension, uint32_t num_blocks) {
  size_t radix_size = num_blocks * (params::degree + 1);
  size_t big_lwe_id = blockIdx.x;
  size_t radix_id = big_lwe_id / num_blocks;
  size_t block_id = big_lwe_id % num_blocks;

  auto cur_res_radix = &result_blocks[radix_id * radix_size];
  auto cur_left_radix = &input_blocks[2 * radix_id * radix_size];
  auto cur_right_radix = &input_blocks[(2 * radix_id + 1) * radix_size];

  Torus *cur_res_block = &cur_res_radix[block_id * (params::degree + 1)];
  Torus *cur_left_block = &cur_left_radix[block_id * (params::degree + 1)];
  Torus *cur_right_block = &cur_right_radix[block_id * (params::degree + 1)];

  size_t tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    cur_res_block[tid] = cur_left_block[tid] + cur_right_block[tid];
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    cur_res_block[params::degree] =
        cur_left_block[params::degree] + cur_right_block[params::degree];
  }
}

template <typename Torus, class params>
__global__ void add_lsb_msb(Torus *result_blocks, Torus *lsb_blocks,
                            Torus *msb_blocks, uint32_t glwe_dimension,
                            uint32_t lsb_count, uint32_t msb_count,
                            uint32_t num_blocks) {
  size_t big_lwe_id = blockIdx.x;
  size_t radix_id = big_lwe_id / num_blocks;
  size_t block_id = big_lwe_id % num_blocks;
  size_t lsb_block_id = block_id - radix_id;
  size_t msb_block_id = block_id - radix_id - 1;

  bool process_lsb = (radix_id <= block_id);
  bool process_msb = (radix_id + 1 <= block_id);

  auto cur_res_ct = &result_blocks[big_lwe_id * (params::degree + 1)];

  Torus *cur_lsb_radix = &lsb_blocks[(2 * num_blocks - radix_id + 1) *
                                     radix_id / 2 * (params::degree + 1)];
  Torus *cur_msb_radix = (process_msb)
                             ? &msb_blocks[(2 * num_blocks - radix_id - 1) *
                                           radix_id / 2 * (params::degree + 1)]
                             : nullptr;
  Torus *cur_lsb_ct = (process_lsb)
                          ? &cur_lsb_radix[lsb_block_id * (params::degree + 1)]
                          : nullptr;
  Torus *cur_msb_ct = (process_msb)
                          ? &cur_msb_radix[msb_block_id * (params::degree + 1)]
                          : nullptr;
  size_t tid = threadIdx.x;

  for (int i = 0; i < params::opt; i++) {
    cur_res_ct[tid] = (process_lsb) ? cur_lsb_ct[tid] : 0;
    cur_res_ct[tid] += (process_msb) ? cur_msb_ct[tid] : 0;
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    cur_res_ct[params::degree] = (process_lsb) ? cur_lsb_ct[params::degree] : 0;
    cur_res_ct[params::degree] +=
        (process_msb) ? cur_msb_ct[params::degree] : 0;
  }
}

// Function will copy and shift blocks by one into lsb and msb
// it works only for big lwe ciphertexts(when dimension is power of 2),
// it only takes (num_blocks + 1) / 2 * num_blocks blocks as an input
// e.g. radix_ct[0] contains num_blocks block, radix_ct[1] contains
// bum_blocks - 1 blocks ... radix_ct[num_blocks - 1] contains 1 block;
// blockIdx.x is the id of block inside the vector of radix_ct
// radix_id = (2 * N + 1 - sqrt((2 * N + 1) * ( 2 * N + 1) - 8 * i)) / 2.;
// local_block_id = i - (2 * N - radix_id + 1) / 2. * radix_id;
// where N = num_blocks and i = blockIdx.x
// local_block_id is id of a block inside a radix_ct
template <typename Torus, class params>
__global__ void copy_and_block_shift_scalar_multiply_add(
    Torus *radix_lwe_left, Torus *lsb_ciphertext, Torus *msb_ciphertext,
    Torus *radix_lwe_right, int num_blocks, int scalar_value) {

  size_t block_id = blockIdx.x;
  double D = sqrt((2 * num_blocks + 1) * (2 * num_blocks + 1) - 8 * block_id);
  size_t radix_id = int((2 * num_blocks + 1 - D) / 2.);
  size_t local_block_id =
      block_id - (2 * num_blocks - radix_id + 1) / 2. * radix_id;
  bool process_msb = (local_block_id < (num_blocks - radix_id - 1));
  auto cur_lsb_block = &lsb_ciphertext[block_id * (params::degree + 1)];
  auto cur_msb_block =
      (process_msb)
          ? &msb_ciphertext[(block_id - radix_id) * (params::degree + 1)]
          : nullptr;
  auto cur_ct_right = &radix_lwe_right[radix_id * (params::degree + 1)];
  auto cur_src = &radix_lwe_left[local_block_id * (params::degree + 1)];
  size_t tid = threadIdx.x;

  for (int i = 0; i < params::opt; i++) {
    Torus value = cur_src[tid] * scalar_value;
    if (process_msb) {
      cur_lsb_block[tid] = cur_msb_block[tid] = value + cur_ct_right[tid];
    } else {
      cur_lsb_block[tid] = value + cur_ct_right[tid];
    }
    tid += params::degree / params::opt;
  }
  if (threadIdx.x == 0) {
    Torus value = cur_src[params::degree] * scalar_value;
    if (process_msb) {
      cur_lsb_block[params::degree] = cur_msb_block[params::degree] =
          value + cur_ct_right[params::degree];
    } else {
      cur_lsb_block[params::degree] = value + cur_ct_right[params::degree];
    }
  }
}

template <typename Torus, class params>
__global__ void
add_carry_to_messsage(Torus *message_blocks, Torus *carry_blocks,
                      uint32_t glwe_dimension, uint32_t radix_count,
                      uint32_t num_blocks) {
  size_t radix_size = num_blocks * (params::degree + 1);
  size_t big_lwe_id = blockIdx.x;
  size_t radix_id = big_lwe_id / num_blocks;
  size_t block_id = big_lwe_id % num_blocks;

  bool process_carry = block_id;

  auto cur_message_radix = &message_blocks[radix_id * radix_size];
  auto cur_carry_radix = &carry_blocks[radix_id * radix_size];

  Torus *cur_message_ct = &cur_message_radix[block_id * (params::degree + 1)];
  Torus *cur_carry_ct =
      (process_carry) ? &cur_carry_radix[(block_id - 1) * (params::degree + 1)]
                      : nullptr;

  size_t tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    cur_message_ct[tid] += (process_carry) ? cur_carry_ct[tid] : 0;
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    cur_message_ct[params::degree] +=
        (process_carry) ? cur_carry_ct[params::degree] : 0;
  }
}

template <typename Torus, typename STorus, class params>
void full_propagate_inplace(
    void *v_stream, uint32_t gpu_index, Torus *input_blocks, Torus *acc_message,
    Torus *acc_carry, Torus *tvi_message, Torus *tvi_carry, Torus *ksk,
    Torus *bsk, Torus *small_lwe_vector, Torus *big_lwe_vector,
    int8_t *pbs_buffer, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t ks_base_log, uint32_t ks_level,
    uint32_t pbs_base_log, uint32_t pbs_level, uint32_t radix_count,
    uint32_t num_blocks, uint32_t message_modulus, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  cuda_keyswitch_lwe_ciphertext_vector(
      v_stream, gpu_index, small_lwe_vector, input_blocks, ksk,
      polynomial_size * glwe_dimension, lwe_dimension, ks_base_log, ks_level,
      radix_count * num_blocks);

  cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
      v_stream, gpu_index, big_lwe_vector, acc_carry, tvi_carry,
      small_lwe_vector, bsk, pbs_buffer, lwe_dimension, glwe_dimension,
      polynomial_size, 3, pbs_base_log, pbs_level, radix_count * num_blocks, 1,
      0, max_shared_memory);

  cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
      v_stream, gpu_index, input_blocks, acc_message, tvi_message,
      small_lwe_vector, bsk, pbs_buffer, lwe_dimension, glwe_dimension,
      polynomial_size, 3, pbs_base_log, pbs_level, radix_count * num_blocks, 1,
      0, max_shared_memory);
  add_carry_to_messsage<Torus, params>
      <<<radix_count * num_blocks, params::degree / params::opt, 0, *stream>>>(
          input_blocks, big_lwe_vector, glwe_dimension, radix_count,
          num_blocks);
}

template <typename Torus, class params>
__host__ void scratch_cuda_integer_mult_radix_ciphertext_kb(
    void *v_stream, uint32_t gpu_index, int_mul_memory<Torus> *mem_ptr,
    uint32_t message_modulus, uint32_t carry_modulus, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t pbs_base_log,
    uint32_t pbs_level, uint32_t ks_base_log, uint32_t ks_level,
    uint32_t num_blocks, PBS_TYPE pbs_type, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {
  cudaSetDevice(gpu_index);
  auto stream = static_cast<cudaStream_t *>(v_stream);

  // 'vector_result_lsb' contains blocks from all possible shifts of
  // radix_lwe_left
  int lsb_vector_block_count = num_blocks * (num_blocks + 1) / 2;

  // 'vector_result_msb' contains blocks from all possible shifts of
  // radix_lwe_left except the last blocks of each shift
  int msb_vector_block_count = num_blocks * (num_blocks - 1) / 2;

  int total_block_count = lsb_vector_block_count + msb_vector_block_count;

  Torus *vector_result_sb = (Torus *)cuda_malloc_async(
      total_block_count * (polynomial_size * glwe_dimension + 1) *
          sizeof(Torus),
      stream, gpu_index);

  Torus *block_mul_res = (Torus *)cuda_malloc_async(
      total_block_count * (polynomial_size * glwe_dimension + 1) *
          sizeof(Torus),
      stream, gpu_index);

  Torus *small_lwe_vector = (Torus *)cuda_malloc_async(
      total_block_count * (lwe_dimension + 1) * sizeof(Torus), stream,
      gpu_index);

  Torus *lwe_pbs_out_array =
      (Torus *)cuda_malloc_async((glwe_dimension * polynomial_size + 1) *
                                     total_block_count * sizeof(Torus),
                                 stream, gpu_index);

  Torus *test_vector_array = (Torus *)cuda_malloc_async(
      2 * (glwe_dimension + 1) * polynomial_size * sizeof(Torus), stream,
      gpu_index);
  Torus *message_acc = (Torus *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(Torus), stream,
      gpu_index);
  Torus *carry_acc = (Torus *)cuda_malloc_async(
      (glwe_dimension + 1) * polynomial_size * sizeof(Torus), stream,
      gpu_index);

  Torus *test_vector_indexes = (Torus *)cuda_malloc_async(
      total_block_count * sizeof(Torus), stream, gpu_index);
  Torus *tvi_message = (Torus *)cuda_malloc_async(
      total_block_count * sizeof(Torus), stream, gpu_index);
  Torus *tvi_carry = (Torus *)cuda_malloc_async(
      total_block_count * sizeof(Torus), stream, gpu_index);

  int8_t *pbs_buffer;

  uint64_t max_buffer_size = 0;
  switch (pbs_type) {
  case MULTI_BIT:
    max_buffer_size = get_max_buffer_size_multibit_bootstrap(lwe_dimension,
        glwe_dimension, polynomial_size, pbs_level, total_block_count);

    if (allocate_gpu_memory)
      pbs_buffer =
          (int8_t *)cuda_malloc_async(max_buffer_size, stream, gpu_index);

    scratch_cuda_multi_bit_pbs_64(
        v_stream, gpu_index, &pbs_buffer, lwe_dimension, glwe_dimension,
        polynomial_size, pbs_level, 3, total_block_count,
        cuda_get_max_shared_memory(gpu_index), false);
    break;
  case LOW_LAT:
    scratch_cuda_bootstrap_low_latency_64(
        v_stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
        pbs_level, total_block_count, cuda_get_max_shared_memory(gpu_index),
        allocate_gpu_memory);
    break;
  case AMORTIZED:
    scratch_cuda_bootstrap_amortized_64(
        stream, gpu_index, &pbs_buffer, glwe_dimension, polynomial_size,
        total_block_count, cuda_get_max_shared_memory(gpu_index),
        allocate_gpu_memory);
    break;
  default:
    break;
  }

  auto lsb_acc = test_vector_array;
  auto msb_acc = &test_vector_array[(glwe_dimension + 1) * polynomial_size];

  generate_lsb_msb_accumulators<Torus, params>(
      v_stream, gpu_index, lsb_acc, msb_acc, message_acc, carry_acc,
      test_vector_indexes, tvi_message, tvi_carry, glwe_dimension,
      message_modulus, carry_modulus, lsb_vector_block_count,
      msb_vector_block_count);

  mem_ptr->vector_result_sb = vector_result_sb;
  mem_ptr->block_mul_res = block_mul_res;
  mem_ptr->small_lwe_vector = small_lwe_vector;
  mem_ptr->lwe_pbs_out_array = lwe_pbs_out_array;
  mem_ptr->test_vector_array = test_vector_array;
  mem_ptr->message_acc = message_acc;
  mem_ptr->carry_acc = carry_acc;
  mem_ptr->test_vector_indexes = test_vector_indexes;
  mem_ptr->tvi_message = tvi_message;
  mem_ptr->tvi_carry = tvi_carry;
  mem_ptr->pbs_buffer = pbs_buffer;
}

template <typename Torus, class params>
__global__ void
prepare_message_and_carry_blocks(Torus *src, Torus *messages, Torus *carries,
                                 uint32_t glwe_dimension, uint32_t r,
                                 uint32_t n, uint32_t f, uint32_t d) {
  int b = blockIdx.x;
  double D =
      sqrt(1. * (2 * n - 2 * f + d) * (2 * n - 2 * f + d) - 4. * d * 2 * b);
  int radix_id = (2. * n - 2 * f + d - D) / (2 * d);

  int result_message_id =
      b - (2 * n - 2 * f - d * (radix_id - 1)) * (radix_id) / 2;
  int result_carry_id = result_message_id;
  int src_block_id = result_message_id + f + d * (radix_id);

  int number_of_blocks_before_cur_message_radix =
      (2 * n - 2 * f - d * (radix_id - 1)) * radix_id / 2;
  int number_of_blocks_before_cur_carry_radix =
      number_of_blocks_before_cur_message_radix - radix_id;

  auto cur_src_radix =
      &src[radix_id * n * (glwe_dimension * params::degree + 1)];
  auto cur_message_radix = &messages[number_of_blocks_before_cur_message_radix *
                                     (glwe_dimension * params::degree + 1)];

  bool process_carry = ((n - f - d * radix_id - 1) > result_carry_id);
  auto cur_carry_radix =
      (process_carry) ? &carries[number_of_blocks_before_cur_carry_radix *
                                 (glwe_dimension * params::degree + 1)]
                      : nullptr;

  auto cur_src_block =
      &cur_src_radix[src_block_id * (glwe_dimension * params::degree + 1)];
  auto cur_message_block =
      &cur_message_radix[result_message_id *
                         (glwe_dimension * params::degree + 1)];
  auto cur_carry_block =
      (process_carry) ? &cur_carry_radix[result_carry_id *
                                         (glwe_dimension * params::degree + 1)]
                      : nullptr;

  int tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    Torus val = cur_src_block[tid];
    cur_message_block[tid] = val;
    if (process_carry)
      cur_carry_block[tid] = val;
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    Torus val = cur_src_block[params::degree];
    cur_message_block[params::degree] = val;
    if (process_carry)
      cur_carry_block[params::degree] = val;
  }
}

template <typename Torus, class params>
__global__ void accumulate_message_and_carry_blocks_optimised(
    Torus *dst, Torus *messages, Torus *carries, uint32_t glwe_dimension,
    uint32_t r, uint32_t n, uint32_t f, uint32_t d) {
  int b = blockIdx.x;
  double D =
      sqrt(1. * (2 * n - 2 * f + d) * (2 * n - 2 * f + d) - 4. * d * 2 * b);
  int radix_id = (2. * n - 2 * f + d - D) / (2 * d);

  int input_message_id =
      b - (2 * n - 2 * f - d * (radix_id - 1)) * (radix_id) / 2;
  int input_carry_id = input_message_id;
  int dst_block_id = input_message_id + f + d * (radix_id);

  int number_of_blocks_before_cur_message_radix =
      (2 * n - 2 * f - d * (radix_id - 1)) * radix_id / 2;
  int number_of_blocks_before_cur_carry_radix =
      number_of_blocks_before_cur_message_radix - radix_id;

  auto cur_dst_radix =
      &dst[radix_id * n * (glwe_dimension * params::degree + 1)];
  auto cur_message_radix = &messages[number_of_blocks_before_cur_message_radix *
                                     (glwe_dimension * params::degree + 1)];

  bool process_carry = ((n - f - d * radix_id - 1) > input_carry_id);
  auto cur_carry_radix =
      (process_carry) ? &carries[number_of_blocks_before_cur_carry_radix *
                                 (glwe_dimension * params::degree + 1)]
                      : nullptr;

  auto cur_dst_block =
      &cur_dst_radix[dst_block_id * (glwe_dimension * params::degree + 1)];
  auto cur_message_block =
      &cur_message_radix[input_message_id *
                         (glwe_dimension * params::degree + 1)];
  auto cur_carry_block =
      (process_carry) ? &cur_carry_radix[input_carry_id *
                                         (glwe_dimension * params::degree + 1)]
                      : nullptr;

  int tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    Torus val = cur_message_block[tid];
    if (process_carry)
      val += cur_carry_block[tid];
    cur_dst_block[tid] += val;
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    Torus val = cur_message_block[params::degree];
    if (process_carry)
      val += cur_carry_block[params::degree];
    cur_dst_block[params::degree] += val;
  }
}

template <typename Torus, typename STorus, class params>
void execute_pbs_single_gpu(
    void *v_stream, uint32_t gpu_index, Torus *lwe_array_out, Torus *lut_vector,
    Torus *lut_vector_indexes, Torus *lwe_array_in, void *bootstrapping_key,
    int8_t *pbs_buffer, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t base_log, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t num_lut_vectors,
    uint32_t lwe_idx, uint32_t max_shared_memory, PBS_TYPE pbs_type) {
  switch (pbs_type) {
  case MULTI_BIT:
    cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
        v_stream, gpu_index, lwe_array_out, lut_vector, lut_vector_indexes,
        lwe_array_in, bootstrapping_key, pbs_buffer, lwe_dimension,
        glwe_dimension, polynomial_size, 3, base_log, level_count,
        input_lwe_ciphertext_count, num_lut_vectors, lwe_idx,
        max_shared_memory);
    break;
  case LOW_LAT:
    cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
        v_stream, gpu_index, lwe_array_out, lut_vector, lut_vector_indexes,
        lwe_array_in, bootstrapping_key, pbs_buffer, lwe_dimension,
        glwe_dimension, polynomial_size, base_log, level_count,
        input_lwe_ciphertext_count, num_lut_vectors, lwe_idx,
        max_shared_memory);
    break;
  case AMORTIZED:
    cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
        v_stream, gpu_index, lwe_array_out, lut_vector, lut_vector_indexes,
        lwe_array_in, bootstrapping_key, pbs_buffer, lwe_dimension,
        glwe_dimension, polynomial_size, base_log, level_count,
        input_lwe_ciphertext_count, num_lut_vectors, lwe_idx,
        max_shared_memory);
    break;
  default:
    break;
  }
}

template <typename Torus, typename STorus, class params>
__host__ void host_integer_mult_radix_kb(
    void *v_stream, uint32_t gpu_index, uint64_t *radix_lwe_out,
    uint64_t *radix_lwe_left, uint64_t *radix_lwe_right,
    uint32_t *ct_degree_out, uint32_t *ct_degree_left,
    uint32_t *ct_degree_right, void *bsk, uint64_t *ksk,
    int_mul_memory<Torus> *mem_ptr, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level,
    uint32_t ks_base_log, uint32_t ks_level, uint32_t num_blocks,
    PBS_TYPE pbs_type, uint32_t max_shared_memory) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  // 'vector_result_lsb' contains blocks from all possible right shifts of
  // radix_lwe_left, only nonzero blocks are kept
  int lsb_vector_block_count = num_blocks * (num_blocks + 1) / 2;

  // 'vector_result_msb' contains blocks from all possible shifts of
  // radix_lwe_left except the last blocks of each shift. Only nonzero blocks
  // are kept
  int msb_vector_block_count = num_blocks * (num_blocks - 1) / 2;

  // total number of blocks msb and lsb
  int total_block_count = lsb_vector_block_count + msb_vector_block_count;

  auto mem = *mem_ptr;

  // buffer to keep all lsb and msb shifts
  // for lsb all nonzero blocks of each right shifts are kept
  // for 0 shift num_blocks blocks
  // for 1 shift num_blocks - 1 blocks
  // for num_blocks - 1 shift 1 block
  // (num_blocks + 1) * num_blocks / 2 blocks
  // for msb we don't keep track for last blocks so
  // for 0 shift num_blocks - 1 blocks
  // for 1 shift num_blocks - 2 blocks
  // for num_blocks - 1 shift  0 blocks
  // (num_blocks - 1) * num_blocks / 2 blocks
  // in total num_blocks^2 blocks
  // in each block three is big polynomial with
  // glwe_dimension * polynomial_size + 1 coefficients
  Torus *vector_result_sb = mem.vector_result_sb;

  // buffer to keep lsb_vector + msb_vector
  // addition will happen in full terms so there will be
  // num_blocks terms and each term will have num_blocks block
  // num_blocks^2 blocks in total
  // and each blocks has big lwe ciphertext with
  // glwe_dimension * polynomial_size + 1 coefficients
  Torus *block_mul_res = mem.block_mul_res;

  // buffer to keep keyswitch result of num_blocks^2 ciphertext
  // in total it has num_blocks^2 small lwe ciphertexts with
  // lwe_dimension +1 coefficients
  Torus *small_lwe_vector = mem.small_lwe_vector;

  // buffer to keep pbs result for num_blocks^2 lwe_ciphertext
  // in total it has num_blocks^2 big lwe ciphertexts with
  // glwe_dimension * polynomial_size + 1 coefficients
  Torus *lwe_pbs_out_array = mem.lwe_pbs_out_array;

  // it contains two test vector, first for lsb extraction,
  // second for msb extraction, with total length =
  // 2 * (glwe_dimension + 1) * polynomial_size
  Torus *test_vector_array = mem.test_vector_array;

  // accumulator to extract message
  // with length (glwe_dimension + 1) * polynomial_size
  Torus *message_acc = mem.message_acc;

  // accumulator to extract carry
  // with length (glwe_dimension + 1) * polynomial_size
  Torus *carry_acc = mem.carry_acc;

  // indexes to refer test_vector for lsb and msb extraction
  // it has num_blocks^2 elements
  // first lsb_vector_block_count element is 0
  // next msb_vector_block_count element is 1
  Torus *test_vector_indexes = mem.test_vector_indexes;

  // indexes to refer test_vector for message extraction
  // it has num_blocks^2 elements and all of them are 0
  Torus *tvi_message = mem.tvi_message;

  // indexes to refer test_vector for carry extraction
  // it has num_blocks^2 elements and all of them are 0
  Torus *tvi_carry = mem.tvi_carry;

  // buffer to calculate num_blocks^2 pbs in parallel
  // used by pbs for intermediate calculation
  int8_t *pbs_buffer = mem.pbs_buffer;

  Torus *vector_result_lsb = &vector_result_sb[0];
  Torus *vector_result_msb =
      &vector_result_sb[lsb_vector_block_count *
                        (polynomial_size * glwe_dimension + 1)];

  dim3 grid(lsb_vector_block_count, 1, 1);
  dim3 thds(params::degree / params::opt, 1, 1);

  copy_and_block_shift_scalar_multiply_add<Torus, params>
      <<<grid, thds, 0, *stream>>>(radix_lwe_left, vector_result_lsb,
                                   vector_result_msb, radix_lwe_right,
                                   num_blocks, 4);

  cuda_keyswitch_lwe_ciphertext_vector(
      v_stream, gpu_index, small_lwe_vector, vector_result_sb, ksk,
      polynomial_size * glwe_dimension, lwe_dimension, ks_base_log, ks_level,
      total_block_count);

  execute_pbs_single_gpu<Torus, STorus, params>(
      v_stream, gpu_index, lwe_pbs_out_array, test_vector_array,
      test_vector_indexes, small_lwe_vector, bsk, pbs_buffer, glwe_dimension,
      lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
      total_block_count, 2, 0, max_shared_memory, pbs_type);

  vector_result_lsb = &lwe_pbs_out_array[0];
  vector_result_msb =
      &lwe_pbs_out_array[lsb_vector_block_count *
                         (polynomial_size * glwe_dimension + 1)];

  add_lsb_msb<Torus, params>
      <<<num_blocks * num_blocks, params::degree / params::opt, 0, *stream>>>(
          block_mul_res, vector_result_lsb, vector_result_msb, glwe_dimension,
          lsb_vector_block_count, msb_vector_block_count, num_blocks);

  auto old_blocks = block_mul_res;
  auto new_blocks = vector_result_sb;

  // position of first nonzero block in first radix
  int f = 1;

  // increment size for first nonzero block position from radix[i] to next
  // radix[i+1]
  int d = 2;

  // amount of current radixes after block_mul
  size_t r = num_blocks;
  while (r > 1) {
    r /= 2;
    tree_add<Torus, params>
        <<<r * num_blocks, params::degree / params::opt, 0, *stream>>>(
            new_blocks, old_blocks, glwe_dimension, num_blocks);

    if (r > 1) {
      int message_blocks_count = (2 * num_blocks - 2 * f - d * r + d) * r / 2;
      int carry_blocks_count = message_blocks_count - r;
      int total_blocks = message_blocks_count + carry_blocks_count;

      auto message_blocks_vector = old_blocks;
      auto carry_blocks_vector =
          &old_blocks[message_blocks_count *
                      (glwe_dimension * polynomial_size + 1)];

      prepare_message_and_carry_blocks<Torus, params>
          <<<message_blocks_count, params::degree / params::opt, 0, *stream>>>(
              new_blocks, message_blocks_vector, carry_blocks_vector,
              glwe_dimension, r, num_blocks, f, d);

      cuda_keyswitch_lwe_ciphertext_vector(
          v_stream, gpu_index, small_lwe_vector, old_blocks, ksk,
          polynomial_size * glwe_dimension, lwe_dimension, ks_base_log,
          ks_level, total_blocks);

      auto small_lwe_message_vector = small_lwe_vector;
      auto small_lwe_carry_vector =
          &small_lwe_vector[message_blocks_count * (lwe_dimension + 1)];

      execute_pbs_single_gpu<Torus, STorus, params>(
          v_stream, gpu_index, message_blocks_vector, message_acc, tvi_message,
          small_lwe_message_vector, bsk, pbs_buffer, glwe_dimension,
          lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
          message_blocks_count, 2, 0, max_shared_memory, pbs_type);

      execute_pbs_single_gpu<Torus, STorus, params>(
          v_stream, gpu_index, carry_blocks_vector, carry_acc, tvi_carry,
          small_lwe_carry_vector, bsk, pbs_buffer, glwe_dimension,
          lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
          carry_blocks_count, 2, 0, max_shared_memory, pbs_type);

      accumulate_message_and_carry_blocks_optimised<Torus, params>
          <<<message_blocks_count, params::degree / params::opt, 0, *stream>>>(
              new_blocks, message_blocks_vector, carry_blocks_vector,
              glwe_dimension, r, num_blocks, f, d);

      f *= 2;
      d *= 2;
    }
    std::swap(new_blocks, old_blocks);
  }

  cudaMemcpyAsync(radix_lwe_out, old_blocks,
                  num_blocks * (glwe_dimension * polynomial_size + 1) *
                      sizeof(Torus),
                  cudaMemcpyDeviceToDevice, *stream);
}

template <typename Torus, class params>
__host__ void scratch_cuda_integer_mult_radix_ciphertext_kb_multi_gpu(
    int_mul_memory<Torus> *mem_ptr, uint64_t *bsk, uint64_t *ksk,
    uint32_t message_modulus, uint32_t carry_modulus, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t pbs_base_log,
    uint32_t pbs_level, uint32_t ks_base_log, uint32_t ks_level,
    uint32_t num_blocks, PBS_TYPE pbs_type, uint32_t max_shared_memory,
    bool allocate_gpu_memory) {

  // printf("Checking for multiple GPUs...\n");

  assert(("Error (integer multiplication multi GPU) application must be built "
          "as a 64-bit target",
          mem_ptr->IsAppBuiltAs64()));

  int gpu_n;
  cudaGetDeviceCount(&gpu_n);
  // printf("CUDA-capable device count: %i\n", gpu_n);

  assert(("Two or more cuda-capable GPUs are required", gpu_n >= 2));

  cudaDeviceProp prop[64];

  for (int i = 0; i < gpu_n; i++) {
    cudaGetDeviceProperties(&prop[i], i);
  }

  // printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  // We find a gpu with highest p2p capable pairs.
  int gpu_peers[gpu_n] = {0};
  int max_peers = 0;
  int max_pees_index = -1;

  // Show all the combinations of supported P2P GPUs

  bool total_p2p_check = true;
  int can_access_peer;
  for (int i = 0; i < gpu_n; i++) {
    for (int j = 0; j < gpu_n; j++) {
      if (i != j) {
        cudaDeviceCanAccessPeer(&can_access_peer, i, j);
        // printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
        // prop[i].name,
        //        i, prop[j].name, j, can_access_peer ? "Yes" : "No");
        if (!can_access_peer) {
          total_p2p_check = false;
        }
      }
    }
  }

  assert(("Error Not all GPU pairs have p2p access", check));

  mem_ptr->p2p_gpu_count = gpu_n;

  for (int i = 1; i < gpu_n; i++) {
    // printf("Enabling peer access between GPU%d and GPU%d...\n", 0, i);
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(i, 0);
    cudaSetDevice(i);
    cudaDeviceEnablePeerAccess(0, 0);
  }

  for (int i = 0; i < gpu_n; i++) {
    cudaSetDevice(i);
    mem_ptr->streams[i] = cuda_create_stream(i);
  }

  cudaSetDevice(0);

  scratch_cuda_integer_mult_radix_ciphertext_kb<Torus, params>(
      (void *)mem_ptr->streams[0], 0, mem_ptr, message_modulus, carry_modulus,
      glwe_dimension, lwe_dimension, polynomial_size, pbs_base_log, pbs_level,
      ks_base_log, ks_level, num_blocks, pbs_type, max_shared_memory, false);

  int lsb_vector_block_count = num_blocks * (num_blocks + 1) / 2;

  int msb_vector_block_count = num_blocks * (num_blocks - 1) / 2;

  int total_block_count = lsb_vector_block_count + msb_vector_block_count;

  assert(gpu_n <= 8);
  for (int i = 0; i < gpu_n; i++) {
    mem_ptr->device_to_device_buffer[i] = (Torus *)cuda_malloc_async(
        total_block_count * (glwe_dimension * polynomial_size + 1) *
            sizeof(Torus),
        mem_ptr->streams[i], 0);
  }

  for (int i = 0; i < gpu_n; i++) {

    cudaSetDevice(i);

    // each gpu will have total_blocks / gpu_n input
    // last gpu will have extra total_blocks % gpu_n
    int block_count = total_block_count / gpu_n;
    if (i == gpu_n - 1) {
      block_count += total_block_count % gpu_n;
    }
    uint64_t max_buffer_size = get_max_buffer_size_multibit_bootstrap
        (lwe_dimension,
        glwe_dimension, polynomial_size, pbs_level, block_count);
    // memory space used by pbs, allocated for each gpu
    int8_t *pbs_buffer =
        (int8_t *)cuda_malloc_async(max_buffer_size, mem_ptr->streams[i], i);

    // buffer for pbs input small lwe ciphertexts for each gpu
    Torus *pbs_input_multi_gpu = (Torus *)cuda_malloc_async(
        block_count * (lwe_dimension + 1) * sizeof(Torus), mem_ptr->streams[i],
        i);

    // buffer for pbs output big lwe ciphertexts for each gpu
    Torus *pbs_output_multi_gpu = (Torus *)cuda_malloc_async(
        block_count * (glwe_dimension * polynomial_size + 1) * sizeof(Torus),
        mem_ptr->streams[i], i);

    // test vector
    // same test vectors will be allocated and copied to all gpus
    // 4 test vectors in total
    // 0 to extract lsb
    // 1 to extract msb
    // 2 to extract message
    // 3 to extract carry
    Torus *test_vector_multi_gpu = (Torus *)cuda_malloc_async(
        (4 * (glwe_dimension + 1) * polynomial_size) * sizeof(Torus),
        mem_ptr->streams[i], i);

    // test vector indexes
    // same values will be copied in all gpus
    // tvi_lsb is filled with 0
    // tvi_msb is filled with 1
    // tvi_message is filled with 2
    // tvi_carry is filled with 3
    Torus *tvi_lsb = (Torus *)cuda_malloc_async(block_count * sizeof(Torus),
                                                mem_ptr->streams[i], i);
    Torus *tvi_msb = (Torus *)cuda_malloc_async(block_count * sizeof(Torus),
                                                mem_ptr->streams[i], i);
    Torus *tvi_message = (Torus *)cuda_malloc_async(block_count * sizeof(Torus),
                                                    mem_ptr->streams[i], i);
    Torus *tvi_carry = (Torus *)cuda_malloc_async(block_count * sizeof(Torus),
                                                  mem_ptr->streams[i], i);

    // execute scratch function and prepare each gpu for pbs and allocate
    // pbs_buffer
    scratch_cuda_multi_bit_pbs_64(mem_ptr->streams[i], i, &pbs_buffer,
                                  lwe_dimension, glwe_dimension,
                                  polynomial_size, pbs_level, 3, block_count,
                                  cuda_get_max_shared_memory(i), false);

    int glwe_length = (glwe_dimension + 1) * polynomial_size;

    // copy lsb and msb test vectors to gpu_i
    cudaMemcpyPeerAsync(test_vector_multi_gpu, i, mem_ptr->test_vector_array, 0,
                        2 * glwe_length * sizeof(Torus),
                        *(mem_ptr->streams[i]));

    // copy message test vector to gpu_i
    cudaMemcpyPeerAsync(&test_vector_multi_gpu[2 * glwe_length], i,
                        mem_ptr->message_acc, 0, glwe_length * sizeof(Torus),
                        *(mem_ptr->streams[i]));

    // copy carry test vector to gpu_i
    cudaMemcpyPeerAsync(&test_vector_multi_gpu[3 * glwe_length], i,
                        mem_ptr->carry_acc, 0, glwe_length * sizeof(Torus),
                        *(mem_ptr->streams[i]));

    // fill test vector indexes with corresponding values
    int block = 256;
    int grid = (block_count + block - 1) / block;

    fill<Torus, params>
        <<<grid, block, 0, *(mem_ptr->streams[i])>>>(tvi_lsb, 0, block_count);
    fill<Torus, params>
        <<<grid, block, 0, *(mem_ptr->streams[i])>>>(tvi_msb, 1, block_count);
    fill<Torus, params><<<grid, block, 0, *(mem_ptr->streams[i])>>>(
        tvi_message, 2, block_count);
    fill<Torus, params>
        <<<grid, block, 0, *(mem_ptr->streams[i])>>>(tvi_carry, 3, block_count);

    // assign pointers to corresponding struct multi gpu members
    mem_ptr->pbs_buffer_multi_gpu[i] = pbs_buffer;
    mem_ptr->pbs_input_multi_gpu[i] = pbs_input_multi_gpu;
    mem_ptr->pbs_output_multi_gpu[i] = pbs_output_multi_gpu;
    mem_ptr->test_vector_multi_gpu[i] = test_vector_multi_gpu;

    mem_ptr->tvi_lsb_multi_gpu[i] = tvi_lsb;
    mem_ptr->tvi_msb_multi_gpu[i] = tvi_msb;
    mem_ptr->tvi_message_multi_gpu[i] = tvi_message;
    mem_ptr->tvi_carry_multi_gpu[i] = tvi_carry;

    // copy bsk and ksk to gpu_i
    // if i == 0, pointer already exists for gpu and no copy is needed
    if (i == 0) {
      mem_ptr->bsk_multi_gpu[i] = bsk;
      mem_ptr->ksk_multi_gpu[i] = ksk;
    } else {
      int bsk_size = (glwe_dimension + 1) * (glwe_dimension + 1) * pbs_level *
                     lwe_dimension * polynomial_size * 8 / 3;
      int ksk_size = ks_level * (lwe_dimension + 1) * polynomial_size;

      uint64_t *cur_bsk = (Torus *)cuda_malloc_async(bsk_size * sizeof(Torus),
                                                     mem_ptr->streams[i], i);
      uint64_t *cur_ksk = (Torus *)cuda_malloc_async(ksk_size * sizeof(Torus),
                                                     mem_ptr->streams[i], i);
      cudaMemcpyPeerAsync(cur_bsk, i, bsk, 0, bsk_size * sizeof(Torus),
                          *(mem_ptr->streams[i]));
      cudaMemcpyPeerAsync(cur_ksk, i, ksk, 0, ksk_size * sizeof(Torus),
                          *(mem_ptr->streams[i]));

      mem_ptr->bsk_multi_gpu[i] = cur_bsk;
      mem_ptr->ksk_multi_gpu[i] = cur_ksk;
    }
  }
}

// Function to apply lookup table,
// It has two mode
//  lsb_msb_mode == true - extracts lsb and msb
//  lsb_msb_mode == false - extracts message and carry
template <typename Torus, typename STorus, class params>
void apply_lookup_table(Torus *input_ciphertexts, Torus *output_ciphertexts,
                        int_mul_memory<Torus> *mem_ptr, uint32_t glwe_dimension,
                        uint32_t lwe_dimension, uint32_t polynomial_size,
                        uint32_t pbs_base_log, uint32_t pbs_level,
                        uint32_t ks_base_log, uint32_t ks_level,
                        uint32_t lsb_message_blocks_count,
                        uint32_t msb_carry_blocks_count,
                        uint32_t max_shared_memory, bool lsb_msb_mode) {

  int total_blocks_count = lsb_message_blocks_count + msb_carry_blocks_count;
  int gpu_n = mem_ptr->p2p_gpu_count;
  if (total_blocks_count < gpu_n)
    gpu_n = total_blocks_count;
  int gpu_blocks_count = total_blocks_count / gpu_n;
  int big_lwe_size = glwe_dimension * polynomial_size + 1;
  int small_lwe_size = lwe_dimension + 1;

#pragma omp parallel for num_threads(gpu_n)
  for (int i = 0; i < gpu_n; i++) {
    cudaSetDevice(i);
    auto this_stream = mem_ptr->streams[i];
    // Index where input and output blocks start for current gpu
    int big_lwe_start_index = i * gpu_blocks_count * big_lwe_size;

    // Last gpu might have extra blocks to process if total blocks number is not
    // divisible by gpu_n
    if (i == gpu_n - 1) {
      gpu_blocks_count += total_blocks_count % gpu_n;
    }

    int can_access_peer;
    cudaDeviceCanAccessPeer(&can_access_peer, i, 0);
    if (i == 0) {
      check_cuda_error(
          cudaMemcpyAsync(mem_ptr->pbs_output_multi_gpu[i],
                          &input_ciphertexts[big_lwe_start_index],
                          gpu_blocks_count * big_lwe_size * sizeof(Torus),
                          cudaMemcpyDeviceToDevice, *this_stream));
    } else if (can_access_peer) {
      check_cuda_error(cudaMemcpyPeerAsync(
          mem_ptr->pbs_output_multi_gpu[i], i,
          &input_ciphertexts[big_lwe_start_index], 0,
          gpu_blocks_count * big_lwe_size * sizeof(Torus), *this_stream));
    } else {
      // Uses host memory as middle ground
      cuda_memcpy_async_to_cpu(mem_ptr->device_to_device_buffer[i],
                               &input_ciphertexts[big_lwe_start_index],
                               gpu_blocks_count * big_lwe_size * sizeof(Torus),
                               this_stream, i);
      cuda_memcpy_async_to_gpu(
          mem_ptr->pbs_output_multi_gpu[i], mem_ptr->device_to_device_buffer[i],
          gpu_blocks_count * big_lwe_size * sizeof(Torus), this_stream, i);
    }

    // when lsb and msb have to be extracted
    //  for first lsb_count blocks we need lsb_acc
    //  for last msb_count blocks we need msb_acc
    // when message and carry have tobe extracted
    //  for first message_count blocks we need message_acc
    //  for last carry_count blocks we need carry_acc
    Torus *cur_tvi;
    if (lsb_msb_mode) {
      cur_tvi = (big_lwe_start_index < lsb_message_blocks_count)
                    ? mem_ptr->tvi_lsb_multi_gpu[i]
                    : mem_ptr->tvi_msb_multi_gpu[i];

    } else {
      cur_tvi = (big_lwe_start_index < lsb_message_blocks_count)
                    ? mem_ptr->tvi_message_multi_gpu[i]
                    : mem_ptr->tvi_carry_multi_gpu[i];
    }

    // execute keyswitch on a current gpu with corresponding input and output
    // blocks pbs_output_multi_gpu[i] is an input for keyswitch and
    // pbs_input_multi_gpu[i] is an output for keyswitch
    cuda_keyswitch_lwe_ciphertext_vector(
        this_stream, i, mem_ptr->pbs_input_multi_gpu[i],
        mem_ptr->pbs_output_multi_gpu[i], mem_ptr->ksk_multi_gpu[i],
        polynomial_size * glwe_dimension, lwe_dimension, ks_base_log, ks_level,
        gpu_blocks_count);

    // execute pbs on a current gpu with corresponding input and output
    cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
        this_stream, i, mem_ptr->pbs_output_multi_gpu[i],
        mem_ptr->test_vector_multi_gpu[i], cur_tvi,
        mem_ptr->pbs_input_multi_gpu[i], mem_ptr->bsk_multi_gpu[i],
        mem_ptr->pbs_buffer_multi_gpu[i], lwe_dimension, glwe_dimension,
        polynomial_size, 3, pbs_base_log, pbs_level, gpu_blocks_count, 2, 0,
        max_shared_memory);

    // lookup table is applied and now data from current gpu have to be copied
    // back to gpu_0 in 'output_ciphertexts' buffer
    if (i == 0) {
      check_cuda_error(
          cudaMemcpyAsync(&output_ciphertexts[big_lwe_start_index],
                          mem_ptr->pbs_output_multi_gpu[i],
                          gpu_blocks_count * big_lwe_size * sizeof(Torus),
                          cudaMemcpyDeviceToDevice, *this_stream));
    } else if (can_access_peer) {
      check_cuda_error(cudaMemcpyPeerAsync(
          &output_ciphertexts[big_lwe_start_index], 0,
          mem_ptr->pbs_output_multi_gpu[i], i,
          gpu_blocks_count * big_lwe_size * sizeof(Torus), *this_stream));
    } else {
      // Uses host memory as middle ground
      cuda_memcpy_async_to_cpu(
          mem_ptr->device_to_device_buffer[i], mem_ptr->pbs_output_multi_gpu[i],
          gpu_blocks_count * big_lwe_size * sizeof(Torus), this_stream, i);
      cuda_memcpy_async_to_gpu(&output_ciphertexts[big_lwe_start_index],
                               mem_ptr->device_to_device_buffer[i],
                               gpu_blocks_count * big_lwe_size * sizeof(Torus),
                               this_stream, i);
    }
  }
}

template <typename Torus, typename STorus, class params>
__host__ void host_integer_mult_radix_kb_multi_gpu(
    uint64_t *radix_lwe_out, uint64_t *radix_lwe_left,
    uint64_t *radix_lwe_right, uint32_t *ct_degree_out,
    uint32_t *ct_degree_left, uint32_t *ct_degree_right, uint64_t *bsk,
    uint64_t *ksk, int_mul_memory<Torus> *mem_ptr, uint32_t message_modulus,
    uint32_t carry_modulus, uint32_t glwe_dimension, uint32_t lwe_dimension,
    uint32_t polynomial_size, uint32_t pbs_base_log, uint32_t pbs_level,
    uint32_t ks_base_log, uint32_t ks_level, uint32_t num_blocks,
    uint32_t max_shared_memory) {

  // 'vector_result_lsb' contains blocks from all possible right shifts of
  // radix_lwe_left, only nonzero blocks are kept
  int lsb_vector_block_count = num_blocks * (num_blocks + 1) / 2;

  // 'vector_result_msb' contains blocks from all possible shifts of
  // radix_lwe_left except the last blocks of each shift. Only nonzero blocks
  // are kept
  int msb_vector_block_count = num_blocks * (num_blocks - 1) / 2;

  // total number of blocks msb and lsb
  int total_block_count = lsb_vector_block_count + msb_vector_block_count;

  auto mem = *mem_ptr;

  Torus *vector_result_sb = mem.vector_result_sb;

  Torus *block_mul_res = mem.block_mul_res;

  // buffer to keep keyswitch result of num_blocks^2 ciphertext
  // in total it has num_blocks^2 small lwe ciphertexts with
  // lwe_dimension +1 coefficients
  Torus *small_lwe_vector = mem.small_lwe_vector;

  // buffer to keep pbs result for num_blocks^2 lwe_ciphertext
  // in total it has num_blocks^2 big lwe ciphertexts with
  // glwe_dimension * polynomial_size + 1 coefficients
  Torus *lwe_pbs_out_array = mem.lwe_pbs_out_array;

  // it contains two test vector, first for lsb extraction,
  // second for msb extraction, with total length =
  // 2 * (glwe_dimension + 1) * polynomial_size
  Torus *test_vector_array = mem.test_vector_array;

  // accumulator to extract message
  // with length (glwe_dimension + 1) * polynomial_size
  Torus *message_acc = mem.message_acc;

  // accumulator to extract carry
  // with length (glwe_dimension + 1) * polynomial_size
  Torus *carry_acc = mem.carry_acc;

  // indexes to refer test_vector for lsb and msb extraction
  // it has num_blocks^2 elements
  // first lsb_vector_block_count element is 0
  // next msb_vector_block_count element is 1
  Torus *test_vector_indexes = mem.test_vector_indexes;

  // indexes to refer test_vector for message extraction
  // it has num_blocks^2 elements and all of them are 0
  Torus *tvi_message = mem.tvi_message;

  // indexes to refer test_vector for carry extraction
  // it has num_blocks^2 elements and all of them are 0
  Torus *tvi_carry = mem.tvi_carry;

  // buffer to calculate num_blocks^2 pbs in parallel
  // used by pbs for intermediate calculation
  int8_t *pbs_buffer = mem.pbs_buffer;

  Torus *vector_result_lsb = &vector_result_sb[0];
  Torus *vector_result_msb =
      &vector_result_sb[lsb_vector_block_count *
                        (polynomial_size * glwe_dimension + 1)];

  dim3 grid(lsb_vector_block_count, 1, 1);
  dim3 thds(params::degree / params::opt, 1, 1);

  auto stream0 = mem_ptr->streams[0];
  int gpu0 = 0;
  cudaSetDevice(0);
  copy_and_block_shift_scalar_multiply_add<Torus, params>
      <<<grid, thds, 0, *stream0>>>(radix_lwe_left, vector_result_lsb,
                                    vector_result_msb, radix_lwe_right,
                                    num_blocks, 4);
  check_cuda_error(cudaGetLastError());

  apply_lookup_table<Torus, STorus, params>(
      vector_result_sb, lwe_pbs_out_array, mem_ptr, glwe_dimension,
      lwe_dimension, polynomial_size, pbs_base_log, pbs_level, ks_base_log,
      ks_level, lsb_vector_block_count, msb_vector_block_count,
      max_shared_memory, true);

  vector_result_lsb = &lwe_pbs_out_array[0];
  vector_result_msb =
      &lwe_pbs_out_array[lsb_vector_block_count *
                         (polynomial_size * glwe_dimension + 1)];

  cudaSetDevice(0);
  add_lsb_msb<Torus, params>
      <<<num_blocks * num_blocks, params::degree / params::opt, 0, *stream0>>>(
          block_mul_res, vector_result_lsb, vector_result_msb, glwe_dimension,
          lsb_vector_block_count, msb_vector_block_count, num_blocks);
  check_cuda_error(cudaGetLastError());
  auto old_blocks = block_mul_res;
  auto new_blocks = vector_result_sb;

  // position of first nonzero block in first radix
  int f = 1;

  // increment size for first nonzero block position from radix[i] to next
  // radix[i+1]
  int d = 2;

  // amount of current radixes after block_mul
  size_t r = num_blocks;
  while (r > 1) {
    cudaSetDevice(0);
    r /= 2;
    tree_add<Torus, params>
        <<<r * num_blocks, params::degree / params::opt, 0, *stream0>>>(
            new_blocks, old_blocks, glwe_dimension, num_blocks);
    check_cuda_error(cudaGetLastError());

    if (r > 1) {
      int message_blocks_count = (2 * num_blocks - 2 * f - d * r + d) * r / 2;
      int carry_blocks_count = message_blocks_count - r;
      int total_blocks = message_blocks_count + carry_blocks_count;

      auto message_blocks_vector = old_blocks;
      auto carry_blocks_vector =
          &old_blocks[message_blocks_count *
                      (glwe_dimension * polynomial_size + 1)];

      prepare_message_and_carry_blocks<Torus, params>
          <<<message_blocks_count, params::degree / params::opt, 0, *stream0>>>(
              new_blocks, message_blocks_vector, carry_blocks_vector,
              glwe_dimension, r, num_blocks, f, d);
      check_cuda_error(cudaGetLastError());

      apply_lookup_table<Torus, STorus, params>(
          old_blocks, old_blocks, mem_ptr, glwe_dimension, lwe_dimension,
          polynomial_size, pbs_base_log, pbs_level, ks_base_log, ks_level,
          message_blocks_count, carry_blocks_count, max_shared_memory, false);

      cudaSetDevice(0);

      accumulate_message_and_carry_blocks_optimised<Torus, params>
          <<<message_blocks_count, params::degree / params::opt, 0, *stream0>>>(
              new_blocks, message_blocks_vector, carry_blocks_vector,
              glwe_dimension, r, num_blocks, f, d);
      check_cuda_error(cudaGetLastError());

      f *= 2;
      d *= 2;
    }
    std::swap(new_blocks, old_blocks);
  }

  check_cuda_error(cudaMemcpyAsync(
      radix_lwe_out, old_blocks,
      num_blocks * (glwe_dimension * polynomial_size + 1) * sizeof(Torus),
      cudaMemcpyDeviceToDevice, *stream0));
}

#endif // CUDA_MULT_H
