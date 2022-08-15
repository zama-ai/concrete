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
      out_allocated, out_aligned, out_offset, out_size, out_stride});
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
    uint64_t ct0_stride, uint64_t *glwe_ct_allocated, uint64_t *glwe_ct_aligned,
    uint64_t glwe_ct_offset, uint64_t glwe_ct_size, uint64_t glwe_ct_stride,
    mlir::concretelang::RuntimeContext *context,
    std::promise<concretelang::clientlib::MemRefDescriptor<1>> promise) {
  CAPI_ASSERT_ERROR(
      fftw_engine_lwe_ciphertext_discarding_bootstrap_u64_raw_ptr_buffers(
          get_fftw_engine(context), get_engine(context),
          get_fftw_fourier_bootstrap_key_u64(context), out_aligned + out_offset,
          ct0_aligned + ct0_offset, glwe_ct_aligned + glwe_ct_offset));
  promise.set_value(concretelang::clientlib::MemRefDescriptor<1>{
      out_allocated, out_aligned, out_offset, out_size, out_stride});
}

void *memref_bootstrap_async_lwe_u64(
    uint64_t *out_allocated, uint64_t *out_aligned, uint64_t out_offset,
    uint64_t out_size, uint64_t out_stride, uint64_t *ct0_allocated,
    uint64_t *ct0_aligned, uint64_t ct0_offset, uint64_t ct0_size,
    uint64_t ct0_stride, uint64_t *glwe_ct_allocated, uint64_t *glwe_ct_aligned,
    uint64_t glwe_ct_offset, uint64_t glwe_ct_size, uint64_t glwe_ct_stride,
    mlir::concretelang::RuntimeContext *context) {
  std::promise<concretelang::clientlib::MemRefDescriptor<1>> promise;
  auto ret = new std::future<concretelang::clientlib::MemRefDescriptor<1>>(
      promise.get_future());
  std::thread offload_thread(
      async_bootstrap, out_allocated, out_aligned, out_offset, out_size,
      out_stride, ct0_allocated, ct0_aligned, ct0_offset, ct0_size, ct0_stride,
      glwe_ct_allocated, glwe_ct_aligned, glwe_ct_offset, glwe_ct_size,
      glwe_ct_stride, context, std::move(promise));
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
