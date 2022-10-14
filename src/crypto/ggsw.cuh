#ifndef CONCRETE_CORE_GGSW_CUH
#define CONCRETE_CORE_GGSW_CUH

template <typename T, typename ST, class params>
__global__ void batch_fft_ggsw_vectors(double2 *dest, T *src) {

  extern __shared__ char sharedmem[];

  double2 *shared_output = (double2 *)sharedmem;

  // Compression
  int offset = blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  int log_2_opt = params::opt >> 1;
#pragma unroll
  for (int i = 0; i < log_2_opt; i++) {
    ST x = src[(2 * tid) + params::opt * offset];
    ST y = src[(2 * tid + 1) + params::opt * offset];
    shared_output[tid].x = x / (double)std::numeric_limits<T>::max();
    shared_output[tid].y = y / (double)std::numeric_limits<T>::max();
    tid += params::degree / params::opt;
  }
  synchronize_threads_in_block();

  // Switch to the FFT space
  NSMFFT_direct<HalfDegree<params>>(shared_output);
  synchronize_threads_in_block();

  correction_direct_fft_inplace<params>(shared_output);
  synchronize_threads_in_block();

  // Write the output to global memory
  tid = threadIdx.x;
  for (int j = 0; j < log_2_opt; j++) {
    dest[tid + (params::opt >> 1) * offset] = shared_output[tid];
    tid += params::degree / params::opt;
  }
}

/**
 * Applies the FFT transform on sequence of GGSW ciphertexts already in the
 * global memory
 */
template <typename T, typename ST, class params>
void batch_fft_ggsw_vector(void *v_stream, double2 *dest, T *src, uint32_t r,
                           uint32_t glwe_dim, uint32_t polynomial_size,
                           uint32_t level_count) {

  auto stream = static_cast<cudaStream_t *>(v_stream);

  int shared_memory_size = sizeof(double) * polynomial_size;

  int total_polynomials = r * (glwe_dim + 1) * (glwe_dim + 1) * level_count;
  int gridSize = total_polynomials;
  int blockSize = polynomial_size / params::opt;

  batch_fft_ggsw_vectors<T, ST, params>
      <<<gridSize, blockSize, shared_memory_size, *stream>>>(dest, src);
  checkCudaErrors(cudaGetLastError());
}

#endif // CONCRETE_CORE_GGSW_CUH
