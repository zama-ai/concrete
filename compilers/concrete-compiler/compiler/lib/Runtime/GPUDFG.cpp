// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <atomic>
#include <cstdarg>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <concretelang/ClientLib/Types.h>
#include <concretelang/Runtime/stream_emulator_api.h>
#include <concretelang/Runtime/wrappers.h>

#ifdef CONCRETELANG_CUDA_SUPPORT
#include "bootstrap.h"
#include "device.h"
#include "keyswitch.h"
#include "linear_algebra.h"

using MemRef2 = concretelang::clientlib::MemRefDescriptor<2>;
using RuntimeContext = mlir::concretelang::RuntimeContext;

namespace mlir {
namespace concretelang {
namespace gpu_dfg {
namespace {

static std::atomic<size_t> next_device = {0};
static size_t num_devices = 0;

struct Void {};
union Param {
  Void _;
  uint32_t val;
};
union Context {
  Void _;
  RuntimeContext *val;
};
static const int32_t host_location = -1;
struct Stream;
struct Dependence;
struct PBS_buffer {
  PBS_buffer(void *stream, uint32_t gpu_idx, uint32_t glwe_dimension,
             uint32_t polynomial_size, uint32_t input_lwe_ciphertext_count)
      : max_pbs_buffer_samples(input_lwe_ciphertext_count),
        glwe_dim(glwe_dimension), poly_size(polynomial_size),
        gpu_stream(stream), gpu_index(gpu_idx) {
    scratch_cuda_bootstrap_amortized_64(
        gpu_stream, gpu_index, &pbs_buffer, glwe_dim, poly_size,
        max_pbs_buffer_samples, cuda_get_max_shared_memory(gpu_index), true);
  }
  ~PBS_buffer() {
    assert(pbs_buffer != nullptr);
    cleanup_cuda_bootstrap_amortized(gpu_stream, gpu_index, &pbs_buffer);
  }
  int8_t *get_pbs_buffer(void *stream, uint32_t gpu_idx,
                         uint32_t glwe_dimension, uint32_t polynomial_size,
                         uint32_t input_lwe_ciphertext_count) {
    assert(glwe_dimension == glwe_dim);
    assert(polynomial_size == poly_size);
    assert(input_lwe_ciphertext_count <= max_pbs_buffer_samples);
    assert(stream == gpu_stream);
    assert(gpu_idx == gpu_index);
    assert(pbs_buffer != nullptr);
    return pbs_buffer;
  }

private:
  int8_t *pbs_buffer;
  uint32_t max_pbs_buffer_samples;
  uint32_t glwe_dim;
  uint32_t poly_size;
  void *gpu_stream;
  uint32_t gpu_index;
};
struct GPU_DFG {
  uint32_t gpu_idx;
  void *gpu_stream;
  GPU_DFG(uint32_t idx) : gpu_idx(idx), pbs_buffer(nullptr) {
    gpu_stream = cuda_create_stream(idx);
  }
  ~GPU_DFG() {
    if (pbs_buffer != nullptr)
      delete pbs_buffer;
    free_streams();
    cuda_destroy_stream((cudaStream_t *)gpu_stream, gpu_idx);
    free_stream_order_dependent_data();
  }
  inline void register_stream(Stream *s) { streams.push_back(s); }
  inline void register_stream_order_dependent_allocation(void *p) {
    to_free_list.push_back(p);
  }
  inline void free_stream_order_dependent_data() {
    for (auto p : to_free_list)
      free(p);
    to_free_list.clear();
  }
  inline int8_t *get_pbs_buffer(uint32_t glwe_dimension,
                                uint32_t polynomial_size,
                                uint32_t input_lwe_ciphertext_count) {
    if (pbs_buffer == nullptr)
      pbs_buffer = new PBS_buffer(gpu_stream, gpu_idx, glwe_dimension,
                                  polynomial_size, input_lwe_ciphertext_count);
    return pbs_buffer->get_pbs_buffer(gpu_stream, gpu_idx, glwe_dimension,
                                      polynomial_size,
                                      input_lwe_ciphertext_count);
  }
  void drop_pbs_buffer() {
    delete pbs_buffer;
    pbs_buffer = nullptr;
  }
  void free_streams();

private:
  std::list<void *> to_free_list;
  std::list<Stream *> streams;
  PBS_buffer *pbs_buffer;
};
struct Dependence {
  int32_t location;
  int32_t rank;
  MemRef2 host_data;
  void *device_data;
  bool onHostReady;
  bool hostAllocated;
  bool used;
  bool read;
  Dependence(int32_t l, int32_t r, MemRef2 &hd, void *dd, bool ohr,
             bool alloc = false)
      : location(l), rank(r), host_data(hd), device_data(dd), onHostReady(ohr),
        hostAllocated(alloc), used(false), read(false) {}
  Dependence(int32_t l, int32_t r, uint64_t val, void *dd, bool ohr,
             bool alloc = false)
      : location(l), rank(r), device_data(dd), onHostReady(ohr),
        hostAllocated(alloc), used(false), read(false) {
    *host_data.aligned = val;
  }
  inline void free_data(GPU_DFG *dfg) {
    if (location >= 0) {
      cuda_drop_async(device_data, (cudaStream_t *)dfg->gpu_stream, location);
    }
    if (onHostReady && host_data.allocated != nullptr && hostAllocated) {
      // As streams are not synchronized aside from the GET operation,
      // we cannot free host-side data until after the synchronization
      // point as it could still be used by an asynchronous operation.
      dfg->register_stream_order_dependent_allocation(host_data.allocated);
    }
    delete (this);
  }
};
struct Process {
  std::vector<Stream *> input_streams;
  std::vector<Stream *> output_streams;
  GPU_DFG *dfg;
  Param level;
  Param base_log;
  Param input_lwe_dim;
  Param output_lwe_dim;
  Param poly_size;
  Param glwe_dim;
  Param precision;
  Param output_size;
  Context ctx;
  void (*fun)(Process *);
  char name[80];
};

static inline void schedule_kernel(Process *p) { p->fun(p); }
struct Stream {
  stream_type type;
  Dependence *dep;
  Process *producer;
  std::vector<Process *> consumers;
  GPU_DFG *dfg;
  Stream(stream_type t)
      : type(t), dep(nullptr), producer(nullptr), dfg(nullptr) {}
  ~Stream() {
    if (dep != nullptr)
      dep->free_data(dfg);
    delete producer;
  }
  void put(Dependence *d) {
    if (type == TS_STREAM_TYPE_X86_TO_TOPO_LSAP) {
      assert(d->onHostReady &&
             "Host-to-device stream should have data initially on host.");
      size_t data_size = sizeof(uint64_t);
      for (int i = 0; i < d->rank; ++i)
        data_size *= d->host_data.sizes[i];
      d->device_data = cuda_malloc_async(
          data_size, (cudaStream_t *)dfg->gpu_stream, dfg->gpu_idx);
      cuda_memcpy_async_to_gpu(
          d->device_data, d->host_data.aligned + d->host_data.offset, data_size,
          (cudaStream_t *)dfg->gpu_stream, dfg->gpu_idx);
      d->location = dfg->gpu_idx;
    }
    if (type == TS_STREAM_TYPE_TOPO_TO_TOPO_LSAP)
      assert(d->location == (int32_t)dfg->gpu_idx &&
             "Data transfers between GPUs not supported yet");
    // TODO: in case of TS_STREAM_TYPE_TOPO_TO_X86_LSAP, we could
    // initiate transfer back to host early here - but need to
    // allocate memory and then copy out again. Tradeoff might be
    // worth testing.

    // If a dependence was already present, schedule deallocation.
    if (dep != nullptr)
      dep->free_data(dfg);
    dep = d;
  }
  void schedule_work() {
    // If there's no producer process for this stream, it is fed by
    // the control program - nothing to do
    if (producer == nullptr) {
      assert(dep != nullptr && "Data missing on control program stream.");
      return;
    }
    // Recursively go up the DFG to check if new data is available
    for (auto s : producer->input_streams)
      s->schedule_work();
    // Check if any of the inputs have changed - and if so recompute
    // this value. Do not recompute if no changes.
    for (auto s : producer->input_streams)
      if (dep == nullptr || s->dep->used == false) {
        schedule_kernel(producer);
        break;
      }
  }
  Dependence *get_on_host(MemRef2 &out, bool has_scheduled = false) {
    if (!has_scheduled)
      schedule_work();
    assert(dep != nullptr && "GET on empty stream not allowed.");
    dep->used = true;
    // If this was already copied to host, copy out
    if (dep->onHostReady) {
      memref_copy_one_rank(dep->host_data.allocated, dep->host_data.aligned,
                           dep->host_data.offset, dep->host_data.sizes[0],
                           dep->host_data.strides[0], out.allocated,
                           out.aligned, out.offset, out.sizes[0],
                           out.strides[0]);
    } else {
      cuda_memcpy_async_to_cpu(out.aligned + out.offset, dep->device_data,
                               out.sizes[0] * sizeof(uint64_t),
                               (cudaStream_t *)dfg->gpu_stream, dep->location);
      cudaStreamSynchronize(*(cudaStream_t *)dfg->gpu_stream);
      // After this synchronization point, all of the host-side
      // allocated memory can be freed as we know all asynchronous
      // operations have finished.
      dfg->free_stream_order_dependent_data();
      dep->onHostReady = true;
    }
    return dep;
  }
  Dependence *get(int32_t location) {
    schedule_work();
    assert(dep != nullptr && "Dependence could not be computed.");
    dep->used = true;
    if (location == host_location) {
      if (dep->onHostReady)
        return dep;
      size_t data_size = sizeof(uint64_t);
      for (int i = 0; i < dep->rank; ++i)
        data_size *= dep->host_data.sizes[i];
      dep->host_data.allocated = dep->host_data.aligned =
          (uint64_t *)malloc(data_size);
      dep->hostAllocated = true;
      get_on_host(dep->host_data, true);
      return dep;
    }
    assert(dep->location == location &&
           "Multi-GPU within the same SDFG not supported");
    return dep;
  }
};

void GPU_DFG::free_streams() {
  streams.sort();
  streams.unique();
  for (auto s : streams)
    delete s;
}

static inline mlir::concretelang::gpu_dfg::Process *
make_process_1_1(void *dfg, void *sin1, void *sout, void (*fun)(Process *)) {
  mlir::concretelang::gpu_dfg::Process *p =
      new mlir::concretelang::gpu_dfg::Process;
  mlir::concretelang::gpu_dfg::Stream *s1 =
      (mlir::concretelang::gpu_dfg::Stream *)sin1;
  mlir::concretelang::gpu_dfg::Stream *so =
      (mlir::concretelang::gpu_dfg::Stream *)sout;
  p->input_streams.push_back(s1);
  p->dfg = (GPU_DFG *)dfg;
  p->fun = fun;
  p->output_streams.push_back(so);
  s1->consumers.push_back(p);
  so->producer = p;
  so->dfg = s1->dfg = (GPU_DFG *)dfg;
  p->dfg->register_stream(s1);
  p->dfg->register_stream(so);
  return p;
}

static inline mlir::concretelang::gpu_dfg::Process *
make_process_2_1(void *dfg, void *sin1, void *sin2, void *sout,
                 void (*fun)(Process *)) {
  mlir::concretelang::gpu_dfg::Process *p =
      new mlir::concretelang::gpu_dfg::Process;
  mlir::concretelang::gpu_dfg::Stream *s1 =
      (mlir::concretelang::gpu_dfg::Stream *)sin1;
  mlir::concretelang::gpu_dfg::Stream *s2 =
      (mlir::concretelang::gpu_dfg::Stream *)sin2;
  mlir::concretelang::gpu_dfg::Stream *so =
      (mlir::concretelang::gpu_dfg::Stream *)sout;
  p->input_streams.push_back(s1);
  p->input_streams.push_back(s2);
  p->dfg = (GPU_DFG *)dfg;
  p->fun = fun;
  p->output_streams.push_back(so);
  s1->consumers.push_back(p);
  s2->consumers.push_back(p);
  so->producer = p;
  so->dfg = s1->dfg = s2->dfg = (GPU_DFG *)dfg;
  p->dfg->register_stream(s1);
  p->dfg->register_stream(s2);
  p->dfg->register_stream(so);
  return p;
}

// Stream emulator processes
void memref_keyswitch_lwe_u64_process(Process *p) {
  Dependence *idep = p->input_streams[0]->get(p->dfg->gpu_idx);
  void *ct0_gpu = idep->device_data;
  void *ksk_gpu = p->ctx.val->get_ksk_gpu(
      p->level.val, p->input_lwe_dim.val, p->output_lwe_dim.val,
      p->dfg->gpu_idx, (cudaStream_t *)p->dfg->gpu_stream);
  MemRef2 out = {0, 0, 0, {p->output_size.val}, {1}};
  void *out_gpu =
      cuda_malloc_async(out.sizes[0] * sizeof(uint64_t),
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  // Schedule the keyswitch kernel on the GPU
  cuda_keyswitch_lwe_ciphertext_vector_64(
      (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx, out_gpu, ct0_gpu,
      ksk_gpu, p->input_lwe_dim.val, p->output_lwe_dim.val, p->base_log.val,
      p->level.val, 1 /*batch size*/);
  Dependence *dep =
      new Dependence((int32_t)p->dfg->gpu_idx, 1, out, out_gpu, false);
  p->output_streams[0]->put(dep);
}

// Construct the glwe accumulator (on CPU) then put it on a stream as
// input to the bootstrap
void memref_bootstrap_glwe_accumulator_process(Process *p) {
  uint64_t glwe_ct_len = p->poly_size.val * (p->glwe_dim.val + 1);
  uint64_t glwe_ct_size = glwe_ct_len * sizeof(uint64_t);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size);
  Dependence *idep = p->input_streams[0]->get(host_location);
  MemRef2 &mtlu = idep->host_data;
  auto tlu = mtlu.aligned + mtlu.offset;

  // Glwe trivial encryption
  for (size_t i = 0; i < p->poly_size.val * p->glwe_dim.val; i++) {
    glwe_ct[i] = 0;
  }
  for (size_t i = 0; i < p->poly_size.val; i++) {
    glwe_ct[p->poly_size.val * p->glwe_dim.val + i] = tlu[i];
  }
  MemRef2 m = {glwe_ct, glwe_ct, 0, {glwe_ct_len}, {1}};
  Dependence *dep = new Dependence(host_location, 1, m, nullptr, true, true);
  p->output_streams[0]->put(dep);
}

void memref_bootstrap_lwe_u64_process(Process *p) {
  assert(p->output_size.val == p->glwe_dim.val * p->poly_size.val + 1);
  uint32_t num_samples = 1; // TODO batching
  void *fbsk_gpu = p->ctx.val->get_bsk_gpu(
      p->input_lwe_dim.val, p->poly_size.val, p->level.val, p->glwe_dim.val,
      p->dfg->gpu_idx, (cudaStream_t *)p->dfg->gpu_stream);
  Dependence *idep0 = p->input_streams[0]->get(p->dfg->gpu_idx);
  Dependence *idep1 = p->input_streams[1]->get(p->dfg->gpu_idx);
  void *ct0_gpu = idep0->device_data;
  void *glwe_ct_gpu = idep1->device_data;

  MemRef2 out = {0, 0, 0, {p->output_size.val}, {1}};
  uint64_t out_batch_size = out.sizes[0];
  void *out_gpu =
      cuda_malloc_async(out_batch_size * sizeof(uint64_t),
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  // Move test vector indexes to the GPU, the test vector indexes is set of 0
  uint32_t num_test_vectors = 1, lwe_idx = 0,
           test_vector_idxes_size = num_samples * sizeof(uint64_t);
  void *test_vector_idxes = malloc(test_vector_idxes_size);
  memset(test_vector_idxes, 0, test_vector_idxes_size);
  void *test_vector_idxes_gpu =
      cuda_malloc_async(test_vector_idxes_size,
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  cuda_memcpy_async_to_gpu(test_vector_idxes_gpu, test_vector_idxes,
                           test_vector_idxes_size,
                           (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  // Schedule the bootstrap kernel on the GPU
  int8_t *pbs_buffer =
      p->dfg->get_pbs_buffer(p->glwe_dim.val, p->poly_size.val, num_samples);
  cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
      (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx, out_gpu, glwe_ct_gpu,
      test_vector_idxes_gpu, ct0_gpu, fbsk_gpu, (int8_t *)pbs_buffer,
      p->input_lwe_dim.val, p->glwe_dim.val, p->poly_size.val, p->base_log.val,
      p->level.val, num_samples, num_test_vectors, lwe_idx,
      cuda_get_max_shared_memory(p->dfg->gpu_idx));
  p->dfg->drop_pbs_buffer();
  cuda_drop_async(test_vector_idxes_gpu, (cudaStream_t *)p->dfg->gpu_stream,
                  p->dfg->gpu_idx);
  Dependence *dep =
      new Dependence((int32_t)p->dfg->gpu_idx, 1, out, out_gpu, false);
  // As streams are not synchronized, we can only free this vector
  // after a later synchronization point where we are guaranteed that
  // this vector is no longer needed.
  p->dfg->register_stream_order_dependent_allocation(test_vector_idxes);
  p->output_streams[0]->put(dep);
}

void memref_add_lwe_ciphertexts_u64_process(Process *p) {
  Dependence *idep0 = p->input_streams[0]->get(p->dfg->gpu_idx);
  Dependence *idep1 = p->input_streams[1]->get(p->dfg->gpu_idx);
  MemRef2 ct0 = idep0->host_data;
  void *out_gpu =
      cuda_malloc_async(ct0.sizes[0] * sizeof(uint64_t),
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  MemRef2 out = {0, 0, 0, {ct0.sizes[0]}, {1}};
  cuda_add_lwe_ciphertext_vector_64(
      (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx, out_gpu,
      idep0->device_data, idep1->device_data,
      /*p->input_lwe_dim.val*/ ct0.sizes[0] - 1, 1 /* num_samples */);
  Dependence *dep = new Dependence(p->dfg->gpu_idx, 1, out, out_gpu, false);
  p->output_streams[0]->put(dep);
}

void memref_add_plaintext_lwe_ciphertext_u64_process(Process *p) {
  Dependence *idep0 = p->input_streams[0]->get(p->dfg->gpu_idx);
  Dependence *idep1 = p->input_streams[1]->get(p->dfg->gpu_idx);
  MemRef2 ct0 = idep0->host_data;
  void *out_gpu =
      cuda_malloc_async(ct0.sizes[0] * sizeof(uint64_t),
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  MemRef2 out = {0, 0, 0, {ct0.sizes[0]}, {1}};
  cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
      (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx, out_gpu,
      idep0->device_data, idep1->device_data,
      /*p->input_lwe_dim.val*/ ct0.sizes[0] - 1, 1 /* num_samples */);

  Dependence *dep = new Dependence(p->dfg->gpu_idx, 1, out, out_gpu, false);
  p->output_streams[0]->put(dep);
}

void memref_mul_cleartext_lwe_ciphertext_u64_process(Process *p) {
  Dependence *idep0 = p->input_streams[0]->get(p->dfg->gpu_idx);
  Dependence *idep1 = p->input_streams[1]->get(p->dfg->gpu_idx);
  MemRef2 ct0 = idep0->host_data;
  void *out_gpu =
      cuda_malloc_async(ct0.sizes[0] * sizeof(uint64_t),
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  MemRef2 out = {0, 0, 0, {ct0.sizes[0]}, {1}};
  cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
      (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx, out_gpu,
      idep0->device_data, idep1->device_data,
      /*p->input_lwe_dim.val*/ ct0.sizes[0] - 1, 1 /* num_samples */);

  Dependence *dep = new Dependence(p->dfg->gpu_idx, 1, out, out_gpu, false);
  p->output_streams[0]->put(dep);
}

void memref_negate_lwe_ciphertext_u64_process(Process *p) {
  Dependence *idep = p->input_streams[0]->get(p->dfg->gpu_idx);
  MemRef2 ct0 = idep->host_data;
  void *out_gpu =
      cuda_malloc_async(ct0.sizes[0] * sizeof(uint64_t),
                        (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx);
  MemRef2 out = {0, 0, 0, {ct0.sizes[0]}, {1}};
  cuda_negate_lwe_ciphertext_vector_64(
      (cudaStream_t *)p->dfg->gpu_stream, p->dfg->gpu_idx, out_gpu,
      idep->device_data,
      /*p->input_lwe_dim.val*/ ct0.sizes[0] - 1, 1 /* num_samples */);
  Dependence *dep = new Dependence(p->dfg->gpu_idx, 1, out, out_gpu, false);
  p->output_streams[0]->put(dep);
}

} // namespace
} // namespace gpu_dfg
} // namespace concretelang
} // namespace mlir

using namespace mlir::concretelang::gpu_dfg;

// Code generation interface
void stream_emulator_make_memref_add_lwe_ciphertexts_u64_process(void *dfg,
                                                                 void *sin1,
                                                                 void *sin2,
                                                                 void *sout) {
  Process *p = make_process_2_1(dfg, sin1, sin2, sout,
                                memref_add_lwe_ciphertexts_u64_process);
  static int count = 0;
  sprintf(p->name, "add_lwe_ciphertexts_%d", count++);
}

void stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  Process *p = make_process_2_1(
      dfg, sin1, sin2, sout, memref_add_plaintext_lwe_ciphertext_u64_process);
  static int count = 0;
  sprintf(p->name, "add_plaintext_lwe_ciphertexts_%d", count++);
}

void stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  Process *p = make_process_2_1(
      dfg, sin1, sin2, sout, memref_mul_cleartext_lwe_ciphertext_u64_process);
  static int count = 0;
  sprintf(p->name, "mul_cleartext_lwe_ciphertexts_%d", count++);
}

void stream_emulator_make_memref_negate_lwe_ciphertext_u64_process(void *dfg,
                                                                   void *sin1,
                                                                   void *sout) {
  Process *p = make_process_1_1(dfg, sin1, sout,
                                memref_negate_lwe_ciphertext_u64_process);
  static int count = 0;
  sprintf(p->name, "negate_lwe_ciphertext_%d", count++);
}

void stream_emulator_make_memref_keyswitch_lwe_u64_process(
    void *dfg, void *sin1, void *sout, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t output_size,
    void *context) {
  Process *p =
      make_process_1_1(dfg, sin1, sout, memref_keyswitch_lwe_u64_process);
  p->level.val = level;
  p->base_log.val = base_log;
  p->input_lwe_dim.val = input_lwe_dim;
  p->output_lwe_dim.val = output_lwe_dim;
  p->output_size.val = output_size;
  p->ctx.val = (RuntimeContext *)context;
  static int count = 0;
  sprintf(p->name, "keyswitch_%d", count++);
}

void stream_emulator_make_memref_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t precision, uint32_t output_size, void *context) {

  // We need to generate two processes: one for building the glwe
  // accumulator and one for the bootstrap (plus a stream to connect).
  void *accu_s =
      stream_emulator_make_memref_stream("", TS_STREAM_TYPE_X86_TO_TOPO_LSAP);
  Process *accu_p = make_process_1_1(dfg, sin2, accu_s,
                                     memref_bootstrap_glwe_accumulator_process);
  accu_p->input_lwe_dim.val = input_lwe_dim;
  accu_p->poly_size.val = poly_size;
  accu_p->level.val = level;
  accu_p->base_log.val = base_log;
  accu_p->glwe_dim.val = glwe_dim;
  accu_p->precision.val = precision;
  accu_p->output_size.val = output_size;
  accu_p->ctx.val = (RuntimeContext *)context;
  static int count_ = 0;
  sprintf(accu_p->name, "glwe_accumulator_%d", count_++);

  Process *p = make_process_2_1(dfg, sin1, accu_s, sout,
                                memref_bootstrap_lwe_u64_process);
  p->input_lwe_dim.val = input_lwe_dim;
  p->poly_size.val = poly_size;
  p->level.val = level;
  p->base_log.val = base_log;
  p->glwe_dim.val = glwe_dim;
  p->precision.val = precision;
  p->output_size.val = output_size;
  p->ctx.val = (RuntimeContext *)context;
  static int count = 0;
  sprintf(p->name, "bootstrap_%d", count++);
}

void *stream_emulator_make_uint64_stream(const char *name, stream_type stype) {
  return (void *)new Stream(stype);
}
void stream_emulator_put_uint64(void *stream, uint64_t e) {
  Stream *s = (Stream *)stream;
  uint64_t *data = (uint64_t *)malloc(sizeof(uint64_t));
  *data = e;
  MemRef2 m = {data, data, 0, {1}, {1}};
  Dependence *dep = new Dependence(host_location, 1, m, nullptr, true, true);
  s->put(dep);
}
uint64_t stream_emulator_get_uint64(void *stream) {
  uint64_t res;
  auto s = (Stream *)stream;
  MemRef2 m = {&res, &res, 0, {1}, {1}};
  s->get_on_host(m);
  return res;
}

void *stream_emulator_make_memref_stream(const char *name, stream_type stype) {
  return (void *)new Stream(stype);
}
void stream_emulator_put_memref(void *stream, uint64_t *allocated,
                                uint64_t *aligned, uint64_t offset,
                                uint64_t size, uint64_t stride) {
  Stream *s = (Stream *)stream;
  MemRef2 m = {allocated, aligned, offset, {size}, {stride}};
  Dependence *dep = new Dependence(host_location, 1, m, nullptr, true);
  s->put(dep);
}
void stream_emulator_get_memref(void *stream, uint64_t *out_allocated,
                                uint64_t *out_aligned, uint64_t out_offset,
                                uint64_t out_size, uint64_t out_stride) {
  MemRef2 mref = {
      out_allocated, out_aligned, out_offset, {out_size}, {out_stride}};
  auto s = (Stream *)stream;
  s->get_on_host(mref);
}

void *stream_emulator_init() {
  int num;
  if (num_devices == 0) {
    assert(cudaGetDeviceCount(&num) == cudaSuccess);
    num_devices = num;
  }
  int device = next_device.fetch_add(1) % num_devices;
  return new GPU_DFG(device);
}
void stream_emulator_run(void *dfg) {}
void stream_emulator_delete(void *dfg) { delete (GPU_DFG *)dfg; }
#endif
