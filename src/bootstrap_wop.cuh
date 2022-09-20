#ifndef WOP_PBS_H
#define WOP_PBS_H

#include "cooperative_groups.h"

#include "../include/helper_cuda.h"
#include "bootstrap.h"
#include "complex/operations.cuh"
#include "crypto/gadget.cuh"
#include "crypto/torus.cuh"
#include "crypto/ggsw.cuh"
#include "fft/bnsmfft.cuh"
#include "fft/smfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/memory.cuh"
#include "utils/timer.cuh"

template <typename T, class params>
__device__ void fft(double2 *output, T *input){
      synchronize_threads_in_block();

      // Reduce the size of the FFT to be performed by storing
      // the real-valued polynomial into a complex polynomial
      real_to_complex_compressed<T, params>(input, output);
      synchronize_threads_in_block();

      // Switch to the FFT space
      NSMFFT_direct<HalfDegree<params>>(output);
      synchronize_threads_in_block();

      correction_direct_fft_inplace<params>(output);
      synchronize_threads_in_block();
}

template <typename T, typename ST, class params>
__device__ void fft(double2 *output, T *input){
      synchronize_threads_in_block();

      // Reduce the size of the FFT to be performed by storing
      // the real-valued polynomial into a complex polynomial
      real_to_complex_compressed<T, ST, params>(input, output);
      synchronize_threads_in_block();

      // Switch to the FFT space
      NSMFFT_direct<HalfDegree<params>>(output);
      synchronize_threads_in_block();

      correction_direct_fft_inplace<params>(output);
      synchronize_threads_in_block();
}

template <class params>
__device__ void ifft_inplace(double2 *data){
    synchronize_threads_in_block();

    correction_inverse_fft_inplace<params>(data);
    synchronize_threads_in_block();

    NSMFFT_inverse<HalfDegree<params>>(data);
    synchronize_threads_in_block();
}

/*
 * Receives an array of GLWE ciphertexts and two indexes to ciphertexts in this array,
 * and an array of GGSW ciphertexts with a index to one ciphertext in it. Compute a CMUX with these
 * operands and writes the output to a particular index of glwe_out.
 *
 * This function needs polynomial_size threads per block.
 *
 * - glwe_out: An array where the result should be written to.
 * - glwe_in: An array where the GLWE inputs are stored.
 * - ggsw_in: An array where the GGSW input is stored. In the fourier domain.
 * - selected_memory: An array to be used for the accumulators. Can be in the shared memory or
 * global memory.
 * - output_idx: The index of the output where the glwe ciphertext should be written.
 * - input_idx1: The index of the first glwe ciphertext we will use.
 * - input_idx2: The index of the second glwe ciphertext we will use.
 * - glwe_dim: This is k.
 * - polynomial_size: size of the polynomials. This is N.
 * - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 * - l_gadget: number of decomposition levels in the gadget matrix (~4)
 * - ggsw_idx: The index of the GGSW we will use.
 */
template <typename Torus, typename STorus, class params>
__device__ void cmux(
    Torus *glwe_out, Torus* glwe_in, double2 *ggsw_in, char *selected_memory,
    uint32_t output_idx, uint32_t input_idx1, uint32_t input_idx2,
    uint32_t glwe_dim, uint32_t polynomial_size, uint32_t base_log, uint32_t l_gadget,
    uint32_t ggsw_idx){

    // Define glwe_sub
    Torus *glwe_sub_mask = (Torus *) selected_memory;
    Torus *glwe_sub_body = (Torus *) glwe_sub_mask + (ptrdiff_t)polynomial_size;

    int16_t *glwe_mask_decomposed = (int16_t *)(glwe_sub_body + polynomial_size);
    int16_t *glwe_body_decomposed =
      (int16_t *)glwe_mask_decomposed + (ptrdiff_t)polynomial_size;

    double2 *mask_res_fft = (double2 *)(glwe_body_decomposed +
                              polynomial_size);
    double2 *body_res_fft =
          (double2 *)mask_res_fft + (ptrdiff_t)polynomial_size / 2;

    double2 *glwe_fft =
        (double2 *)body_res_fft + (ptrdiff_t)(polynomial_size / 2);

    GadgetMatrix<Torus, params> gadget(base_log, l_gadget);

    /////////////////////////////////////

    // glwe2-glwe1

    // Copy m0 to shared memory to preserve data
    auto m0_mask = &glwe_in[input_idx1 * (glwe_dim + 1) * polynomial_size];
    auto m0_body = m0_mask + polynomial_size;

    // Just gets the pointer for m1 on global memory
    auto m1_mask = &glwe_in[input_idx2 * (glwe_dim + 1) * polynomial_size];
    auto m1_body = m1_mask + polynomial_size;

    // Mask
    sub_polynomial<Torus, params>(
        glwe_sub_mask, m1_mask, m0_mask
    );
    // Body
    sub_polynomial<Torus, params>(
        glwe_sub_body, m1_body, m0_body
    );

    synchronize_threads_in_block();

    // Initialize the polynomial multiplication via FFT arrays
    // The polynomial multiplications happens at the block level
    // and each thread handles two or more coefficients
    int pos = threadIdx.x;
    for (int j = 0; j < params::opt / 2; j++) {
      mask_res_fft[pos].x = 0;
      mask_res_fft[pos].y = 0;
      body_res_fft[pos].x = 0;
      body_res_fft[pos].y = 0;
      pos += params::degree / params::opt;
    }

    // Subtract each glwe operand, decompose the resulting
    // polynomial coefficients to multiply each decomposed level
    // with the corresponding part of the LUT
    for (int decomp_level = 0; decomp_level < l_gadget; decomp_level++) {

      // Decomposition
      gadget.decompose_one_level(glwe_mask_decomposed,
                                 glwe_sub_mask,
                                 decomp_level);
      gadget.decompose_one_level(glwe_body_decomposed,
                                 glwe_sub_body,
                                 decomp_level);

      // First, perform the polynomial multiplication for the mask
      synchronize_threads_in_block();
      fft<int16_t, params>(glwe_fft, glwe_mask_decomposed);

      // External product and accumulate
      // Get the piece necessary for the multiplication
      auto mask_fourier = get_ith_mask_kth_block(
              ggsw_in, ggsw_idx, 0, decomp_level,
              polynomial_size, glwe_dim, l_gadget);
      auto body_fourier = get_ith_body_kth_block(
              ggsw_in, ggsw_idx, 0, decomp_level,
              polynomial_size, glwe_dim, l_gadget);

      synchronize_threads_in_block();

      // Perform the coefficient-wise product
      synchronize_threads_in_block();
      polynomial_product_accumulate_in_fourier_domain<params, double2>(
          mask_res_fft, glwe_fft, mask_fourier);
      polynomial_product_accumulate_in_fourier_domain<params, double2>(
          body_res_fft, glwe_fft, body_fourier);

      // Now handle the polynomial multiplication for the body
      // in the same way
      synchronize_threads_in_block();
      fft<int16_t, params>(glwe_fft, glwe_body_decomposed);

      // External product and accumulate
      // Get the piece necessary for the multiplication
      mask_fourier = get_ith_mask_kth_block(
              ggsw_in, ggsw_idx, 1, decomp_level,
              polynomial_size, glwe_dim, l_gadget);
      body_fourier = get_ith_body_kth_block(
              ggsw_in, ggsw_idx, 1, decomp_level,
              polynomial_size, glwe_dim, l_gadget);

      synchronize_threads_in_block();

      polynomial_product_accumulate_in_fourier_domain<params, double2>(
          mask_res_fft, glwe_fft, mask_fourier);
      polynomial_product_accumulate_in_fourier_domain<params, double2>(
          body_res_fft, glwe_fft, body_fourier);

    }

    // IFFT
    synchronize_threads_in_block();
    ifft_inplace<params>(mask_res_fft);
    ifft_inplace<params>(body_res_fft);
    synchronize_threads_in_block();

    // Write the output
    Torus *mb_mask = &glwe_out[output_idx * (glwe_dim + 1) * polynomial_size];
    Torus *mb_body = mb_mask + polynomial_size;

    int tid = threadIdx.x;
    for(int i = 0; i < params::opt; i++){
        mb_mask[tid] = m0_mask[tid];
        mb_body[tid] = m0_body[tid];
        tid += params::degree / params::opt;
    }

    add_to_torus<Torus, params>(mask_res_fft, mb_mask);
    add_to_torus<Torus, params>(body_res_fft, mb_body);
}

/**
 * Computes several CMUXes using an array of GLWE ciphertexts and a single GGSW ciphertext.
 * The GLWE ciphertexts are picked two-by-two in sequence. Each thread block computes a single CMUX.
 *
 * - glwe_out: An array where the result should be written to.
 * - glwe_in: An array where the GLWE inputs are stored.
 * - ggsw_in: An array where the GGSW input is stored. In the fourier domain.
 * - device_mem: An pointer for the global memory in case the shared memory is not big enough to
 * store the accumulators.
 * - device_memory_size_per_block: Memory size needed to store all accumulators for a single block.
 * - glwe_dim: This is k.
 * - polynomial_size: size of the polynomials. This is N.
 * - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 * - l_gadget: number of decomposition levels in the gadget matrix (~4)
 * - ggsw_idx: The index of the GGSW we will use.
 */
template <typename Torus, typename STorus, class params, sharedMemDegree SMD>
__global__ void device_batch_cmux(
    Torus *glwe_out, Torus* glwe_in, double2 *ggsw_in,
    char *device_mem, size_t device_memory_size_per_block,
    uint32_t glwe_dim, uint32_t polynomial_size, uint32_t base_log, uint32_t l_gadget,
    uint32_t ggsw_idx){

    int cmux_idx = blockIdx.x;
    int output_idx = cmux_idx;
    int input_idx1 = (cmux_idx << 1);
    int input_idx2 = (cmux_idx << 1) + 1;

    // We use shared memory for intermediate result
    extern __shared__ char sharedmem[];
    char *selected_memory;

    if constexpr (SMD == FULLSM)
        selected_memory = sharedmem;
    else
        selected_memory = &device_mem[blockIdx.x * device_memory_size_per_block];

    cmux<Torus, STorus, params>(
            glwe_out, glwe_in, ggsw_in,
            selected_memory,
            output_idx, input_idx1, input_idx2,
            glwe_dim, polynomial_size,
            base_log, l_gadget,
            ggsw_idx);

}
/*
 * This kernel executes the CMUX tree used by the hybrid packing of the WoPBS.
 *
 * Uses shared memory for intermediate results
 *
 *  - v_stream: The CUDA stream that should be used.
 *  - glwe_out: A device array for the output GLWE ciphertext.
 *  - ggsw_in: A device array for the GGSW ciphertexts used in each layer.
 *  - lut_vector: A device array for the GLWE ciphertexts used in the first layer.
 * -  polynomial_size: size of the polynomials. This is N.
 *  - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 *  - l_gadget: number of decomposition levels in the gadget matrix (~4)
 *  - r: Number of layers in the tree.
 */
template <typename Torus, typename STorus, class params>
void host_cmux_tree(
        void *v_stream,
        Torus *glwe_out,
        Torus *ggsw_in,
        Torus *lut_vector,
        uint32_t glwe_dimension,
        uint32_t polynomial_size,
        uint32_t base_log,
        uint32_t l_gadget,
        uint32_t r,
        uint32_t max_shared_memory) {

    assert(glwe_dimension == 1); // For larger k we will need to adjust the mask size
    assert(r >= 1);

    auto stream = static_cast<cudaStream_t *>(v_stream);
    int num_lut = (1<<r);

    cuda_initialize_twiddles(polynomial_size, 0);

    int memory_needed_per_block =
      sizeof(Torus) * polynomial_size +   // glwe_sub_mask
      sizeof(Torus) * polynomial_size +   // glwe_sub_body
      sizeof(int16_t) * polynomial_size +   // glwe_mask_decomposed
      sizeof(int16_t) * polynomial_size +   // glwe_body_decomposed
      sizeof(double2) * polynomial_size/2 +   // mask_res_fft
      sizeof(double2) * polynomial_size/2 +   // body_res_fft
      sizeof(double2) * polynomial_size/2;   // glwe_fft

    dim3 thds(polynomial_size / params::opt, 1, 1);

    //////////////////////
//    std::cout << "Applying the FFT on m^tree" << std::endl;
    double2 *d_ggsw_fft_in;
    int ggsw_size = r * polynomial_size * (glwe_dimension + 1) * (glwe_dimension + 1) * l_gadget;
    checkCudaErrors(cudaMalloc((void **)&d_ggsw_fft_in, ggsw_size * sizeof(double)));

    batch_fft_ggsw_vector<Torus, STorus, params>(
            d_ggsw_fft_in, ggsw_in, r, glwe_dimension, polynomial_size, l_gadget);

    //////////////////////

    // Allocate global memory in case parameters are too large
    char *d_mem;
    if (max_shared_memory < memory_needed_per_block) {
        checkCudaErrors(cudaMalloc((void **) &d_mem, memory_needed_per_block * (1 << (r - 1))));
    }else{
        checkCudaErrors(cudaFuncSetAttribute(
            device_batch_cmux<Torus, STorus, params, FULLSM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            memory_needed_per_block));
        // TODO (Agnes): is this necessary?
        checkCudaErrors(cudaFuncSetCacheConfig(
            device_batch_cmux<Torus, STorus, params, FULLSM>,
            cudaFuncCachePreferShared));
    }

    // Allocate buffers
    int glwe_size = (glwe_dimension + 1) * polynomial_size;
    Torus *d_buffer1, *d_buffer2;
    checkCudaErrors(cudaMalloc((void **)&d_buffer1, num_lut * glwe_size * sizeof(Torus)));
    checkCudaErrors(cudaMalloc((void **)&d_buffer2, num_lut * glwe_size * sizeof(Torus)));
    checkCudaErrors(cudaMemcpyAsync(
            d_buffer1, lut_vector,
            num_lut * glwe_size * sizeof(Torus),
            cudaMemcpyDeviceToDevice, *stream));

    Torus *output;
    // Run the cmux tree
    for(int layer_idx = 0; layer_idx < r; layer_idx++){
        output = (layer_idx % 2? d_buffer1 : d_buffer2);
        Torus *input = (layer_idx % 2? d_buffer2 : d_buffer1);

        int num_cmuxes = (1<<(r-1-layer_idx));
        dim3 grid(num_cmuxes, 1, 1);

        // walks horizontally through the leafs
        if(max_shared_memory < memory_needed_per_block)
            device_batch_cmux<Torus, STorus, params, NOSM>
            <<<grid, thds, memory_needed_per_block, *stream>>>(
                    output, input, d_ggsw_fft_in,
                    d_mem, memory_needed_per_block,
                    glwe_dimension, // k
                    polynomial_size, base_log, l_gadget,
                    layer_idx // r
            );
        else
            device_batch_cmux<Torus, STorus, params, FULLSM>
            <<<grid, thds, memory_needed_per_block, *stream>>>(
                    output, input, d_ggsw_fft_in,
                    d_mem, memory_needed_per_block,
                    glwe_dimension, // k
                    polynomial_size, base_log, l_gadget,
                    layer_idx // r
            );

    }

    checkCudaErrors(cudaStreamSynchronize(*stream));
    checkCudaErrors(cudaMemcpy(
            glwe_out, output,
            (glwe_dimension+1) * polynomial_size * sizeof(Torus),
            cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

    // Free memory
   checkCudaErrors(cudaFree(d_ggsw_fft_in));
   checkCudaErrors(cudaFree(d_buffer1));
   checkCudaErrors(cudaFree(d_buffer2));
   if(max_shared_memory < memory_needed_per_block)
       checkCudaErrors(cudaFree(d_mem));

}
#endif //WO_PBS_H
