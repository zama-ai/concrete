// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/Types.h"
#include "concretelang/Runtime/wrappers.h"
#include <assert.h>
#include <future>
#include <stdio.h>
#include <string.h>
#include <thread>

void async_keyswitch(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, mlir::concretelang::RuntimeContext *context,
    std::promise<concretelang::clientlib::MemRefDescriptor<1>> promise) {
  CAPI_ASSERT_ERROR(
      default_engine_discard_keyswitch_lwe_ciphertext_u64_raw_ptr_buffers(
          get_engine(context), get_keyswitch_key_u64(context),
          out_aligned + out_offset, ct0_aligned + ct0_offset));
  promise.set_value(concretelang::clientlib::MemRefDescriptor<1>{
      out_allocated, out_aligned, out_offset, {out_size}, {out_stride}});
}

void *memref_keyswitch_async_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, mlir::concretelang::RuntimeContext *context) {
  std::promise<concretelang::clientlib::MemRefDescriptor<1>> promise;
  auto ret = new std::future<concretelang::clientlib::MemRefDescriptor<1>>(
      promise.get_future());
  std::thread offload_thread(async_keyswitch, out_allocated, out_aligned,
                             out_offset, out_size, out_stride, ct0_allocated,
                             ct0_aligned, ct0_offset, ct0_size, ct0_stride,
                             context, std::move(promise));
  offload_thread.detach();
  return (void *)ret;
}

void async_bootstrap(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context,
    std::promise<concretelang::clientlib::MemRefDescriptor<1>> promise) {

  uint64_t glwe_ct_size = poly_size * (glwe_dim + 1);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size * sizeof(uint64_t));

  std::vector<uint64_t> expanded_tabulated_function_array(poly_size);

  encode_and_expand_lut(expanded_tabulated_function_array.data(), poly_size,
                        precision, tlu_aligned + tlu_offset, tlu_size);

  CAPI_ASSERT_ERROR(
      default_engine_discard_trivially_encrypt_glwe_ciphertext_u64_raw_ptr_buffers(
          get_engine(context), glwe_ct, glwe_ct_size,
          expanded_tabulated_function_array.data(), poly_size));

  CAPI_ASSERT_ERROR(
      fft_engine_lwe_ciphertext_discarding_bootstrap_u64_raw_ptr_buffers(
          get_fft_engine(context), get_engine(context),
          get_fft_fourier_bootstrap_key_u64(context), out_aligned + out_offset,
          ct0_aligned + ct0_offset, glwe_ct));
  promise.set_value(concretelang::clientlib::MemRefDescriptor<1>{
      out_allocated, out_aligned, out_offset, {out_size}, {out_stride}});
  free(glwe_ct);
}

void *memref_bootstrap_async_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *tlu_allocated, uint64_t *tlu_aligned,
    uint64_t tlu_offset, uint64_t tlu_size, uint64_t tlu_stride,
    uint32_t input_lwe_dim, uint32_t poly_size, uint32_t level,
    uint32_t base_log, uint32_t glwe_dim, uint32_t precision,
    mlir::concretelang::RuntimeContext *context) {
  std::promise<concretelang::clientlib::MemRefDescriptor<1>> promise;
  auto ret = new std::future<concretelang::clientlib::MemRefDescriptor<1>>(
      promise.get_future());
  std::thread offload_thread(
      async_bootstrap, out_allocated, out_aligned, out_offset, out_size,
      out_stride, ct0_allocated, ct0_aligned, ct0_offset, ct0_size, ct0_stride,
      tlu_allocated, tlu_aligned, tlu_offset, tlu_size, tlu_stride,
      input_lwe_dim, poly_size, level, base_log, glwe_dim, precision, context,
      std::move(promise));
  offload_thread.detach();
  return (void *)ret;
}

void memref_await_future(uint64_t *out_allocated, uint64_t *out_aligned,
                         uint64_t out_offset, uint64_t out_size,
                         uint64_t out_stride, void *fut, uint64_t *in_allocated,
                         uint64_t *in_aligned, uint64_t in_offset,
                         uint64_t in_size, uint64_t in_stride) {
  auto future =
      static_cast<std::future<concretelang::clientlib::MemRefDescriptor<1>> *>(
          fut);
  auto desc = future->get();
  memref_copy_one_rank(desc.allocated, desc.aligned, desc.offset, desc.sizes[0],
                       desc.strides[0], out_allocated, out_aligned, out_offset,
                       out_size, out_stride);
  delete future;
}
