// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <err.h>
#include <hwloc.h>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <concretelang/Runtime/stream_emulator_api.h>
#include <concretelang/Runtime/wrappers.h>

#ifdef CONCRETELANG_CUDA_SUPPORT
#include "bootstrap.h"
#include "device.h"
#include "keyswitch.h"
#include "linear_algebra.h"

using RuntimeContext = mlir::concretelang::RuntimeContext;

namespace mlir {
namespace concretelang {
namespace gpu_dfg {
namespace {

using MemRef2 = MemRefDescriptor<2>;

// When not using all accelerators on the machine, we distribute work
// by assigning the default accelerator for each SDFG to next_device
// round-robin.
static std::atomic<size_t> next_device = {0};

// Resources available (or set as requested by user through
// environment variables) on the machine. Defaults to using all
// available resources.
static size_t num_devices = 0;            // Set SDFG_NUM_GPUS to configure
static size_t num_cores = 1;              // Set SDFG_NUM_THREADS to configure
static size_t device_compute_factor = 16; // Set SDFG_DEVICE_TO_CORE_RATIO
// How much more memory than just input size is required on GPU to execute
static float gpu_memory_inflation_factor = 1.5;

// Get the byte size of a rank 2 MemRef
static inline size_t memref_get_data_size(MemRef2 &m) {
  return m.sizes[0] * m.sizes[1] * sizeof(uint64_t);
}

// Copy contiguous rank 2 MemRef 'in' to 'out'
static inline void memref_copy_contiguous(MemRef2 &out, MemRef2 &in) {
  assert(in.sizes[0] == out.sizes[0] && in.sizes[1] == out.sizes[1] &&
         "memref_copy_contiguous sizes differ");
  assert(in.strides[0] == out.strides[0] && in.strides[1] == out.strides[1] &&
         "memref_copy_contiguous strides differ");
  assert(in.strides[0] == in.sizes[1] && in.strides[1] == 1 &&
         "memref_copy_contiguous strides not compatible with contiguous "
         "storage.");
  memcpy(out.aligned + out.offset, in.aligned + in.offset,
         memref_get_data_size(in));
}

// Copy contiguous rank 2 MemRef to a newly allocated, returned MemRef
static inline MemRef2 memref_copy_alloc(MemRef2 &m) {
  uint64_t *data = (uint64_t *)malloc(memref_get_data_size(m));
  MemRef2 ret = {
      data, data, 0, {m.sizes[0], m.sizes[1]}, {m.strides[0], m.strides[1]}};
  memref_copy_contiguous(ret, m);
  return ret;
}

// Parameter storage for KS and BS processes
struct Void {};
union Param {
  Void _;
  uint32_t val;
};
union Context {
  Void _;
  RuntimeContext *val;
};
// Tracking locations and state for dependences: location is either an
// integer >= 0 corresponding to the device/accelerator index on the
// machine or -1/-2 when it is only available on the host or is split
// across multiple locations.
static const int32_t host_location = -1;
static const int32_t split_location = -2;
// Similarly dependence chunks are either indexed (which does not
// always correlate to the device index on which they are located) or
// this dependence is split further.
static const int32_t single_chunk = -1;
static const int32_t split_chunks = -2;
struct Stream;
struct Dependence;
// Track buffer/scratchpad for the PBS to avoid re-allocating it on
// the device. Reuse where possible or reallocate if a larger buffer
// is required.
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
    cleanup_cuda_bootstrap_amortized(gpu_stream, gpu_index, &pbs_buffer);
  }
  int8_t *get_pbs_buffer(void *stream, uint32_t gpu_idx,
                         uint32_t glwe_dimension, uint32_t polynomial_size,
                         uint32_t input_lwe_ciphertext_count) {
    assert(glwe_dimension <= glwe_dim);
    assert(polynomial_size <= poly_size);
    assert(input_lwe_ciphertext_count <= max_pbs_buffer_samples);
    assert(stream == gpu_stream);
    assert(gpu_idx == gpu_index);
    return pbs_buffer;
  }
  uint32_t get_max_pbs_buffer_samples() { return max_pbs_buffer_samples; }

  int8_t *pbs_buffer;
  uint32_t max_pbs_buffer_samples;
  uint32_t glwe_dim;
  uint32_t poly_size;
  void *gpu_stream;
  uint32_t gpu_index;
};

// Keep track of the GPU/CUDA streams used for each accelerator and
// associated PBS buffer.
struct GPU_state {
  uint32_t gpu_idx;
  void *gpu_stream;
  PBS_buffer *pbs_buffer;
  GPU_state(uint32_t idx)
      : gpu_idx(idx), gpu_stream(nullptr), pbs_buffer(nullptr) {}
  ~GPU_state() {
    if (pbs_buffer != nullptr)
      delete pbs_buffer;
    if (gpu_stream != nullptr)
      cuda_destroy_stream((cudaStream_t *)gpu_stream, gpu_idx);
  }
  inline int8_t *get_pbs_buffer(uint32_t glwe_dimension,
                                uint32_t polynomial_size,
                                uint32_t input_lwe_ciphertext_count) {
    if (pbs_buffer != nullptr && (pbs_buffer->glwe_dim != glwe_dimension ||
                                  pbs_buffer->poly_size != polynomial_size ||
                                  pbs_buffer->get_max_pbs_buffer_samples() <
                                      input_lwe_ciphertext_count)) {
      delete pbs_buffer;
      pbs_buffer = nullptr;
    }
    if (pbs_buffer == nullptr)
      pbs_buffer = new PBS_buffer(get_gpu_stream(), gpu_idx, glwe_dimension,
                                  polynomial_size, input_lwe_ciphertext_count);
    return pbs_buffer->get_pbs_buffer(get_gpu_stream(), gpu_idx, glwe_dimension,
                                      polynomial_size,
                                      input_lwe_ciphertext_count);
  }
  inline void *get_gpu_stream() {
    if (gpu_stream == nullptr)
      gpu_stream = cuda_create_stream(gpu_idx);
    return gpu_stream;
  }
};

// Track resources required for the execution of a single DFG,
// including the GPU states of devices involved in its execution,
// streams and memory allocated which depends on execution progress on
// accelerators before it can be freed.  As execution on accelerators
// is asynchronous, this must wait for the next synchronization point.
struct GPU_DFG {
  std::vector<GPU_state> gpus;
  uint32_t gpu_idx;
  void *gpu_stream;
  GPU_DFG(uint32_t idx) : gpu_idx(idx), pbs_buffer(nullptr) {
    for (uint32_t i = 0; i < num_devices; ++i)
      gpus.push_back(std::move(GPU_state(i)));
    gpu_stream = gpus[idx].get_gpu_stream();
  }
  ~GPU_DFG() {
    free_streams();
    free_stream_order_dependent_data();
  }
  inline void register_stream(Stream *s) { streams.push_back(s); }
  inline void register_stream_order_dependent_allocation(void *p) {
    std::lock_guard<std::mutex> guard(free_list_guard);
    to_free_list.push_back(p);
  }
  inline void free_stream_order_dependent_data() {
    std::lock_guard<std::mutex> guard(free_list_guard);
    for (auto p : to_free_list)
      free(p);
    to_free_list.clear();
  }
  inline int8_t *get_pbs_buffer(uint32_t glwe_dimension,
                                uint32_t polynomial_size,
                                uint32_t input_lwe_ciphertext_count) {
    if (pbs_buffer == nullptr) {
      int8_t *ret = gpus[gpu_idx].get_pbs_buffer(
          glwe_dimension, polynomial_size, input_lwe_ciphertext_count);
      pbs_buffer = gpus[gpu_idx].pbs_buffer;
      return ret;
    }
    return pbs_buffer->get_pbs_buffer(gpu_stream, gpu_idx, glwe_dimension,
                                      polynomial_size,
                                      input_lwe_ciphertext_count);
  }
  void free_streams() {
    streams.sort();
    streams.unique();
    for (auto s : streams)
      delete s;
  }
  inline void *get_gpu_stream(int32_t loc) {
    if (loc < 0)
      return nullptr;
    return gpus[loc].get_gpu_stream();
  }

private:
  std::list<void *> to_free_list;
  std::mutex free_list_guard;
  std::list<Stream *> streams;
  PBS_buffer *pbs_buffer;
};

struct Dependence;
static void sdfg_gpu_debug_print_mref(const char *c, MemRef2 m);
static MemRef2 sdfg_gpu_debug_dependence(Dependence *d, cudaStream_t *s);
static bool sdfg_gpu_debug_compare_memref(MemRef2 &a, MemRef2 &b,
                                          char const *msg);

// Dependences track the location and state of each block of memory
// used as input/output to processes to allow either moving it on/off
// devices or determining when deallocation is possible.
struct Dependence {
  int32_t location;
  MemRef2 host_data;
  void *device_data;
  bool onHostReady;
  bool hostAllocated;
  int32_t chunk_id;
  size_t stream_generation;
  std::vector<Dependence *> chunks;
  Dependence(int32_t l, MemRef2 hd, void *dd, bool ohr, bool alloc = false,
             int32_t chunk_id = single_chunk, size_t gen = 0)
      : location(l), host_data(hd), device_data(dd), onHostReady(ohr),
        hostAllocated(alloc), chunk_id(chunk_id), stream_generation(gen) {}
  Dependence(int32_t l, uint64_t val, void *dd, bool ohr, bool alloc = false,
             int32_t chunk_id = single_chunk, size_t gen = 0)
      : location(l), device_data(dd), onHostReady(ohr), hostAllocated(alloc),
        chunk_id(chunk_id), stream_generation(gen) {
    *host_data.aligned = val;
  }
  // Split a dependence into a number of chunks either to run on
  // multiple GPUs or execute concurrently on the host.
  void split_dependence(size_t num_chunks, size_t num_gpu_chunks,
                        size_t chunk_dim, bool constant) {
    // If this dependence is already split, check that the split
    // matches the new request
    if (chunk_id == split_chunks) {
      if (num_chunks + num_gpu_chunks != chunks.size())
        warnx("WARNING: requesting to split dependence across different number "
              "of chunks (%lu) than it already is split (%lu) which would "
              "require remapping. This is not supported.",
              num_chunks + num_gpu_chunks, chunks.size());
      assert(num_chunks + num_gpu_chunks == chunks.size());
      return;
    }
    if (!chunks.empty()) {
      for (auto c : chunks)
        delete c;
      chunks.clear();
    }
    chunks.resize(num_chunks + num_gpu_chunks, nullptr);

    assert(onHostReady && "Cannot split dependences located on a device.");
    size_t num_samples = host_data.sizes[chunk_dim];
    assert(num_samples > 0);
    // If this is a constant (same data for each chunk), then copy a
    // descriptor corresponding to the whole dependence for each
    // chunk.
    if (constant) {
      for (size_t i = 0; i < num_chunks + num_gpu_chunks; ++i) {
        MemRef2 m = host_data;
        m.allocated = nullptr;
        chunks[i] = new Dependence(host_location, m, nullptr, onHostReady,
                                   false, i, stream_generation);
      }
      return;
    }
    size_t chunk_size =
        num_samples / (num_chunks + num_gpu_chunks * device_compute_factor);
    size_t gpu_chunk_size = chunk_size * device_compute_factor;
    chunk_size = (num_samples - gpu_chunk_size * num_gpu_chunks) / num_chunks;
    size_t chunk_remainder =
        (num_samples - gpu_chunk_size * num_gpu_chunks) % num_chunks;
    uint64_t offset = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t chunk_size_ = (i < chunk_remainder) ? chunk_size + 1 : chunk_size;
      MemRef2 m = host_data;
      m.sizes[chunk_dim] = chunk_size_;
      m.offset = offset + host_data.offset;
      void *dd = (device_data == nullptr) ? device_data
                                          : (uint64_t *)device_data + offset;
      offset += chunk_size_ * host_data.strides[chunk_dim];
      chunks[i] = new Dependence(location, m, dd, onHostReady, false, i,
                                 stream_generation);
    }
    for (size_t i = num_chunks; i < num_chunks + num_gpu_chunks; ++i) {
      MemRef2 m = host_data;
      m.sizes[chunk_dim] = gpu_chunk_size;
      m.offset = offset + host_data.offset;
      void *dd = (device_data == nullptr) ? device_data
                                          : (uint64_t *)device_data + offset;
      offset += gpu_chunk_size * host_data.strides[chunk_dim];
      chunks[i] = new Dependence(location, m, dd, onHostReady, false, i,
                                 stream_generation);
    }
    chunk_id = split_chunks;
    location = split_location;
  }
  void merge_dependence(GPU_DFG *dfg) {
    assert(!chunks.empty() && "Cannot merge dependence with no chunks");
    size_t data_size = 0;
    size_t num_samples = 0;
    for (auto c : chunks) {
      data_size += memref_get_data_size(c->host_data);
      num_samples += c->host_data.sizes[0];
    }
    uint64_t *data = (uint64_t *)malloc(data_size);
    MemRef2 output = {data,
                      data,
                      0,
                      {num_samples, chunks.front()->host_data.sizes[1]},
                      {chunks.front()->host_data.sizes[1], 1}};

    std::list<cudaStream_t *> custreams_used;
    for (auto c : chunks) {
      // Write out the piece in the final target dependence
      size_t csize = memref_get_data_size(c->host_data);
      if (c->onHostReady) {
        memcpy(((char *)output.aligned) + output.offset, c->host_data.aligned,
               csize);
      } else {
        assert(c->location > host_location);
        cudaStream_t *s = (cudaStream_t *)dfg->get_gpu_stream(c->location);
        cuda_memcpy_async_to_cpu(((char *)output.aligned) + output.offset,
                                 c->device_data, csize, s, c->location);
        custreams_used.push_back(s);
      }
      output.offset += csize;
    }
    output.offset = 0;
    for (auto c : chunks)
      c->free_data(dfg, true);
    chunks.clear();

    custreams_used.sort();
    custreams_used.unique();
    for (auto s : custreams_used)
      cudaStreamSynchronize(*s);

    location = host_location;
    onHostReady = true;
    assert(host_data.allocated == nullptr);
    host_data = output;
    assert(device_data == nullptr);
    hostAllocated = true;
    chunk_id = single_chunk;
  }
  void move_chunk_off_device(int32_t chunk_id, GPU_DFG *dfg) {
    chunks[chunk_id]->copy(host_location, dfg);
    cuda_drop_async(
        chunks[chunk_id]->device_data,
        (cudaStream_t *)dfg->get_gpu_stream(chunks[chunk_id]->location),
        chunks[chunk_id]->location);
    chunks[chunk_id]->location = host_location;
  }
  void free_chunk_host_data(int32_t chunk_id, GPU_DFG *dfg) {
    assert(chunks[chunk_id]->location == host_location &&
           chunks[chunk_id]->onHostReady && chunks[chunk_id]->hostAllocated);
    free(chunks[chunk_id]->host_data.allocated);
    chunks[chunk_id]->host_data.allocated = nullptr;
    chunks[chunk_id]->hostAllocated = false;
    chunks[chunk_id]->onHostReady = false;
  }
  void free_chunk_device_data(int32_t chunk_id, GPU_DFG *dfg) {
    assert(chunks[chunk_id]->location > host_location &&
           chunks[chunk_id]->device_data != nullptr);
    cuda_drop_async(
        chunks[chunk_id]->device_data,
        (cudaStream_t *)dfg->get_gpu_stream(chunks[chunk_id]->location),
        chunks[chunk_id]->location);
    chunks[chunk_id]->device_data = nullptr;
  }
  inline void free_data(GPU_DFG *dfg, bool immediate = false) {
    if (location >= 0 && device_data != nullptr) {
      cuda_drop_async(device_data,
                      (cudaStream_t *)dfg->get_gpu_stream(location), location);
    }
    if (onHostReady && host_data.allocated != nullptr && hostAllocated) {
      // As streams are not synchronized aside from the GET operation,
      // we cannot free host-side data until after the synchronization
      // point as it could still be used by an asynchronous operation.
      if (immediate)
        free(host_data.allocated);
      else
        dfg->register_stream_order_dependent_allocation(host_data.allocated);
    }
    for (auto c : chunks)
      c->free_data(dfg, immediate);
    chunks.clear();
    delete (this);
  }
  inline void copy(int32_t loc, GPU_DFG *dfg) {
    size_t data_size = memref_get_data_size(host_data);
    if (loc == location)
      return;
    if (loc == host_location) {
      if (onHostReady)
        return;
      if (host_data.allocated == nullptr) {
        host_data.allocated = host_data.aligned = (uint64_t *)malloc(data_size);
        hostAllocated = true;
      }
      cudaStream_t *s = (cudaStream_t *)dfg->get_gpu_stream(location);
      cuda_memcpy_async_to_cpu(host_data.aligned, device_data, data_size, s,
                               location);
      cudaStreamSynchronize(*s);
      onHostReady = true;
    } else {
      assert(onHostReady &&
             "Device-to-device data transfers not supported yet.");
      cudaStream_t *s = (cudaStream_t *)dfg->get_gpu_stream(loc);
      if (device_data != nullptr)
        cuda_drop_async(device_data, s, location);
      device_data = cuda_malloc_async(data_size, s, loc);
      cuda_memcpy_async_to_gpu(
          device_data, host_data.aligned + host_data.offset, data_size, s, loc);
      location = loc;
    }
  }
};

// Set of input/output streams required to execute a process'
// activation, along with any parameters (for KS / BS) and the
// associated work-function.
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
  Param sk_index;
  Param output_size;
  Context ctx;
  void (*fun)(Process *, int32_t, int32_t, uint64_t *);
  char name[80];
  bool batched_process;
};

void memref_keyswitch_lwe_u64_process(Process *p, int32_t loc, int32_t chunk_id,
                                      uint64_t *out_ptr);
void memref_bootstrap_lwe_u64_process(Process *p, int32_t loc, int32_t chunk_id,
                                      uint64_t *out_ptr);
void memref_add_lwe_ciphertexts_u64_process(Process *p, int32_t loc,
                                            int32_t chunk_id,
                                            uint64_t *out_ptr);
void memref_add_plaintext_lwe_ciphertext_u64_process(Process *p, int32_t loc,
                                                     int32_t chunk_id,
                                                     uint64_t *out_ptr);
void memref_mul_cleartext_lwe_ciphertext_u64_process(Process *p, int32_t loc,
                                                     int32_t chunk_id,
                                                     uint64_t *out_ptr);
void memref_negate_lwe_ciphertext_u64_process(Process *p, int32_t loc,
                                              int32_t chunk_id,
                                              uint64_t *out_ptr);
static inline void schedule_kernel(Process *p, int32_t loc, int32_t chunk_id,
                                   uint64_t *out_ptr) {
  p->fun(p, loc, chunk_id, out_ptr);
}
struct Stream {
  Dependence *dep;
  Dependence *saved_dependence;
  stream_type type;
  Process *producer;
  std::vector<Process *> consumers;
  GPU_DFG *dfg;
  bool batched_stream;
  bool const_stream; // When a batched op uses the same value on this
                     // stream for all ops
  bool ct_stream;
  bool pt_stream;
  size_t generation;
  const char *name;
  Stream(stream_type t, const char *sname = nullptr)
      : dep(nullptr), type(t), producer(nullptr), dfg(nullptr),
        batched_stream(false), const_stream(false), ct_stream(false),
        pt_stream(false), generation(0) {
    if (sname == nullptr) {
      static unsigned long stream_id = 0;
      char *n = new char[16];
      snprintf(n, 16, "stream%llu", stream_id++);
      name = n;
    } else {
      name = sname;
    }
  }
  ~Stream() {
    if (dep != nullptr)
      dep->free_data(dfg);
    if (producer != nullptr)
      delete producer;
  }
  void put(Dependence *d, int32_t chunk_id = single_chunk) {
    assert(chunk_id != split_chunks);
    // If this is a chunk of a larger dependence, then insert it in its
    // position.
    if (chunk_id != single_chunk) {
      // We must always have a dependence present and its split into
      // chunks pre-determined here.
      assert(dep != nullptr);
      if (dep->chunks[chunk_id] != nullptr)
        dep->chunks[chunk_id]->free_data(dfg, true);
      assert(dep->chunks.size() > chunk_id);
      dep->chunks[chunk_id] = d;
    } else {
      //  If a dependence was already present, schedule deallocation.
      if (dep != nullptr)
        dep->free_data(dfg);
      dep = d;
    }
    dep->stream_generation = generation;
  }
  // For a given dependence, traverse the DFG backwards to extract the lattice
  // of kernels required to execute to produce this data
  void extract_producing_graph(std::list<Process *> &queue) {
    // If there's no producer process for this stream, it is fed by
    // the control program - nothing to do
    if (producer == nullptr) {
      assert(dep != nullptr && "Data missing on control program stream.");
      return;
    }
    // Recursively go up the DFG to check if new data is available
    for (auto s : producer->input_streams)
      s->extract_producing_graph(queue);
    // Check if any of the inputs have changed - and if so recompute
    // this value. Do not recompute if no changes.
    size_t input_generation = 0;
    for (auto s : producer->input_streams)
      input_generation += s->generation;
    if (generation < input_generation) {
      queue.push_back(producer);
      generation = input_generation;
    }
  }
  void schedule_work(MemRef2 &out) {
    std::list<Process *> queue;
    extract_producing_graph(queue);

    // Determine if this subgrah bootstraps
    bool is_batched_subgraph = false;
    size_t subgraph_bootstraps = 0;
    for (auto p : queue) {
      is_batched_subgraph |= p->batched_process;
      subgraph_bootstraps +=
          (p->fun == memref_bootstrap_lwe_u64_process) ? 1 : 0;
    }
    // If this subgraph is not batched, then use this DFG's allocated
    // GPU to offload to.  If this does not bootstrap, just execute on
    // the host.
    if (!is_batched_subgraph) {
      for (auto p : queue) {
        schedule_kernel(
            p, (subgraph_bootstraps > 0) ? dfg->gpu_idx : host_location,
            single_chunk, nullptr);
      }
      return;
    }

    // Identify all inputs to these processes that are not also
    // outputs (i.e. data not produced within this subgraph) - and
    // outputs of the subgraph.
    std::list<Stream *> inputs;
    std::list<Stream *> inputs_all;
    std::list<Stream *> outputs;
    std::list<Stream *> intermediate_values;
    for (auto p : queue) {
      inputs.insert(inputs.end(), p->input_streams.begin(),
                    p->input_streams.end());
      outputs.insert(outputs.end(), p->output_streams.begin(),
                     p->output_streams.end());
    }
    inputs.sort();
    inputs.unique();
    inputs_all = inputs;
    outputs.sort();
    outputs.unique();
    intermediate_values = outputs;
    for (auto o : outputs)
      inputs.remove(o);
    for (auto i : inputs_all)
      outputs.remove(i);
    for (auto o : outputs)
      intermediate_values.remove(o);

    assert(!inputs.empty() && !outputs.empty());
    // For now operations have a single output so this subgraph should
    // also only have one output as only operations required to
    // produce this value are taken.
    assert(outputs.size() == 1);

    // Decide on number of chunks to split -- TODO: refine this
    size_t mem_per_sample = 0;
    size_t const_mem_per_sample = 0;
    size_t num_samples = 1;
    size_t num_real_inputs = 0;
    // Only the sizes of inputs is known ahead of execution
    for (auto s : inputs) {
      //  Const streams data is required in whole for each computation,
      //  we treat this separately as it can be substantial.
      if (s->const_stream) {
        const_mem_per_sample += memref_get_data_size(s->dep->host_data);
        continue;
      }
      // If this is a ciphertext
      if (s->ct_stream) {
        mem_per_sample += s->dep->host_data.sizes[1] * sizeof(uint64_t);
        num_real_inputs++;
        if (s->dep->host_data.sizes[0] > num_samples)
          num_samples = s->dep->host_data.sizes[0];
        if (!s->dep->chunks.empty())
          num_samples = s->dep->chunks.size();
      } else {
        mem_per_sample += sizeof(uint64_t);
      }
    }
    // Approximate the memory required for intermediate values and outputs
    mem_per_sample += mem_per_sample *
                      (outputs.size() + intermediate_values.size()) /
                      (num_real_inputs ? num_real_inputs : 1);
    size_t num_chunks = 1;
    size_t num_gpu_chunks = 0;
    int32_t num_devices_to_use = 0;
    // If the subgraph does not have sufficient computational
    // intensity (which we approximate by whether it bootstraps), then
    // we assume (TODO: confirm with profiling) that it is not
    // beneficial to offload to GPU.
    if (subgraph_bootstraps) {
      // Determine maximum GPU granulariry
      size_t gpu_free_mem;
      size_t gpu_total_mem;
      // TODO: this could be improved
      // Force deallocation with a synchronization point
      for (size_t g = 0; g < num_devices; ++g)
        cudaStreamSynchronize(*(cudaStream_t *)dfg->get_gpu_stream(g));
      auto status = cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
      assert(status == cudaSuccess);
      // TODO - for now assume each device on the system has roughly same
      // available memory.
      size_t available_mem = gpu_free_mem;
      // Further assume (TODO) that kernel execution requires some
      // magic factor more meory per sample to execute
      size_t max_samples_per_chunk =
          (available_mem - const_mem_per_sample) /
          ((mem_per_sample ? mem_per_sample : 1) * gpu_memory_inflation_factor);

      if (num_samples < num_cores + device_compute_factor * num_devices) {
        num_devices_to_use = 0;
        num_chunks = std::min(num_cores, num_samples);
      } else {
        num_devices_to_use = num_devices;
        size_t compute_resources =
            num_cores + num_devices * device_compute_factor;
        size_t gpu_chunk_size =
            std::ceil((double)num_samples / compute_resources) *
            device_compute_factor;
        size_t scale_factor =
            std::ceil((double)gpu_chunk_size / max_samples_per_chunk);
        num_chunks = num_cores * scale_factor;
        num_gpu_chunks = num_devices * scale_factor;
      }
    } else {
      num_chunks = std::min(num_cores, num_samples);
    }

    for (auto i : inputs)
      i->dep->split_dependence(num_chunks, num_gpu_chunks,
                               (i->ct_stream) ? 0 : 1, i->const_stream);
    for (auto iv : intermediate_values) {
      if (iv->need_new_gen()) {
        iv->put(new Dependence(split_location,
                               {nullptr, nullptr, 0, {0, 0}, {0, 0}}, nullptr,
                               false, false, split_chunks));
        iv->dep->chunks.resize(num_chunks + num_gpu_chunks, nullptr);
      }
    }
    for (auto o : outputs) {
      if (o->need_new_gen()) {
        o->put(new Dependence(split_location,
                              {nullptr, nullptr, 0, {0, 0}, {0, 0}}, nullptr,
                              false, false, split_chunks));
        o->dep->chunks.resize(num_chunks + num_gpu_chunks, nullptr);
      }
    }

    // Execute graph
    std::list<std::thread> workers;
    std::list<std::thread> gpu_schedulers;
    std::vector<std::list<size_t>> gpu_chunk_list;
    gpu_chunk_list.resize(num_devices);
    int32_t dev = 0;
    for (size_t c = 0; c < num_chunks + num_gpu_chunks; ++c) {
      if (!subgraph_bootstraps) {
        workers.push_back(std::thread(
            [&](std::list<Process *> queue, size_t c, int32_t host_location) {
              for (auto p : queue)
                schedule_kernel(p, host_location, c, nullptr);
              for (auto iv : intermediate_values)
                if (iv->consumers.size() == 1)
                  iv->dep->free_chunk_host_data(c, dfg);
            },
            queue, c, host_location));
      } else {
        if (c < num_chunks) {
          workers.push_back(std::thread(
              [&](std::list<Process *> queue, size_t c, int32_t host_location) {
                for (auto p : queue) {
                  schedule_kernel(p, host_location, c, nullptr);
                }
                for (auto iv : intermediate_values)
                  if (iv->consumers.size() == 1)
                    iv->dep->free_chunk_host_data(c, dfg);
              },
              queue, c, host_location));
        } else {
          // Gather per-device chunk list
          gpu_chunk_list[dev++ % num_devices].push_back(c);
        }
      }
    }
    for (dev = 0; dev < num_devices; ++dev) {
      gpu_schedulers.push_back(std::thread(
          [&](std::list<Process *> queue, int32_t dev) {
            for (size_t c : gpu_chunk_list[dev]) {
              size_t gpu_free_mem;
              size_t gpu_total_mem;
              auto status = cudaSetDevice(dev);
              assert(status == cudaSuccess);
              cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
              assert(status == cudaSuccess);
              for (auto p : queue)
                schedule_kernel(p, dev, c, nullptr);
              for (auto iv : intermediate_values)
                if (iv->consumers.size() > 1)
                  iv->dep->move_chunk_off_device(c, dfg);
                else
                  iv->dep->free_chunk_device_data(c, dfg);
              for (auto o : outputs)
                o->dep->move_chunk_off_device(c, dfg);
              cudaStreamSynchronize(*(cudaStream_t *)dfg->get_gpu_stream(dev));
            }
          },
          queue, dev));
    }
    for (auto &w : workers)
      w.join();
    workers.clear();
    for (auto &gs : gpu_schedulers)
      gs.join();
    gpu_schedulers.clear();
    // Build output out of the separate chunks processed
    for (auto o : outputs) {
      assert(o->batched_stream && o->ct_stream &&
             "Only operations with ciphertext output supported.");
      o->dep->merge_dependence(dfg);
    }
    // We will assume that only one subgraph is being processed per
    // DFG at a time, so we can safely free these here.
    dfg->free_stream_order_dependent_data();
    return;
  }
  Dependence *get_on_host(MemRef2 &out) {
    schedule_work(out);
    assert(dep != nullptr && "GET on empty stream not allowed.");
    // If this was already copied to host, copy out
    if (dep->onHostReady) {
      memref_copy_contiguous(out, dep->host_data);
      return dep;
    } else if (dep->location == split_location) {
      char *pos = (char *)(out.aligned + out.offset);
      std::list<int32_t> devices_used;
      for (auto c : dep->chunks) {
        size_t data_size = memref_get_data_size(c->host_data);
        cuda_memcpy_async_to_cpu(
            pos, c->device_data, data_size,
            (cudaStream_t *)dfg->get_gpu_stream(c->location), c->location);
        pos += data_size;
        devices_used.push_back(c->location);
      }
      // We should only synchronize devices that had data chunks
      devices_used.sort();
      devices_used.unique();
      for (auto i : devices_used)
        cudaStreamSynchronize(*(cudaStream_t *)dfg->get_gpu_stream(i));
    } else {
      size_t data_size = memref_get_data_size(dep->host_data);
      cuda_memcpy_async_to_cpu(out.aligned + out.offset, dep->device_data,
                               data_size, (cudaStream_t *)dfg->gpu_stream,
                               dep->location);
      cudaStreamSynchronize(*(cudaStream_t *)dfg->gpu_stream);
    }
    // After this synchronization point, all of the host-side
    // allocated memory can be freed as we know all asynchronous
    // operations have finished.
    dfg->free_stream_order_dependent_data();
    if (!dep->hostAllocated)
      dep->host_data = memref_copy_alloc(out);
    dep->onHostReady = true;
    dep->hostAllocated = true;
    return dep;
  }
  Dependence *get(int32_t location, int32_t chunk_id = single_chunk) {
    assert(dep != nullptr && "Dependence could not be computed.");
    assert(chunk_id != split_chunks);
    if (chunk_id != single_chunk) {
      Dependence *d = dep->chunks[chunk_id];
      d->copy(location, dfg);
      return d;
    }
    dep->copy(location, dfg);
    return dep;
  }
  inline bool need_new_gen(int32_t chunk_id = single_chunk) {
    if (dep == nullptr)
      return true;
    if (chunk_id == single_chunk && dep->stream_generation != generation)
      return true;
    if (chunk_id != single_chunk) {
      assert(chunk_id < dep->chunks.size());
      if (dep->chunks[chunk_id] == nullptr)
        return true;
      if (dep->chunks[chunk_id]->stream_generation != generation)
        return true;
    }
    return false;
  }
};

static inline mlir::concretelang::gpu_dfg::Process *
make_process_1_1(void *dfg, void *sin1, void *sout,
                 void (*fun)(Process *, int32_t, int32_t, uint64_t *)) {
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
  p->batched_process = s1->batched_stream;
  return p;
}

static inline mlir::concretelang::gpu_dfg::Process *
make_process_2_1(void *dfg, void *sin1, void *sin2, void *sout,
                 void (*fun)(Process *, int32_t, int32_t, uint64_t *)) {
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
  p->batched_process = s1->batched_stream;
  return p;
}

[[maybe_unused]] static void sdfg_gpu_debug_print_mref(const char *c,
                                                       MemRef2 m) {
  std::cout << c << " : " << m.allocated << ", " << m.aligned << ", "
            << m.offset << ", [" << m.sizes[0] << ", " << m.sizes[1] << "], ["
            << m.strides[0] << ", " << m.strides[1] << "]\n";
}

[[maybe_unused]] static MemRef2 sdfg_gpu_debug_dependence(Dependence *d,
                                                          cudaStream_t *s) {
  if (d->onHostReady)
    return d->host_data;
  size_t data_size = memref_get_data_size(d->host_data);
  uint64_t *data = (uint64_t *)malloc(data_size);
  MemRef2 ret = {data,
                 data,
                 0,
                 {d->host_data.sizes[0], d->host_data.sizes[1]},
                 {d->host_data.strides[0], d->host_data.strides[1]}};
  cuda_memcpy_async_to_cpu(data, d->device_data, data_size, s, d->location);
  cudaStreamSynchronize(*s);
  return ret;
}

[[maybe_unused]] static bool
sdfg_gpu_debug_compare_memref(MemRef2 &a, MemRef2 &b, char const *msg) {
  if (a.sizes[0] != b.sizes[0] || a.sizes[1] != b.sizes[1] ||
      a.strides[0] != b.strides[0] || a.strides[1] != b.strides[1])
    return false;
  size_t data_size = memref_get_data_size(a);
  int num = 0;
  for (size_t i = 0; i < data_size / sizeof(uint64_t); ++i)
    if ((a.aligned + a.offset)[i] != (b.aligned + b.offset)[i]) {
      std::cout << msg << " - memrefs differ at position " << i << " "
                << (a.aligned + a.offset)[i] << " " << (b.aligned + b.offset)[i]
                << "\n";
      return false;
    }
  return true;
}

// Stream emulator processes
void memref_keyswitch_lwe_u64_process(Process *p, int32_t loc, int32_t chunk_id,
                                      uint64_t *out_ptr) {
  auto sched = [&](Dependence *d) {
    uint64_t num_samples = d->host_data.sizes[0];
    MemRef2 out = {
        0, 0, 0, {num_samples, p->output_size.val}, {p->output_size.val, 1}};
    size_t data_size = memref_get_data_size(out);
    if (loc == host_location) {
      // If it is not profitable to offload, schedule kernel on CPU
      out.allocated = out.aligned =
          (uint64_t *)((out_ptr != nullptr) ? out_ptr : malloc(data_size));
      memref_batched_keyswitch_lwe_u64(
          out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
          out.strides[0], out.strides[1], d->host_data.allocated,
          d->host_data.aligned, d->host_data.offset, d->host_data.sizes[0],
          d->host_data.sizes[1], d->host_data.strides[0],
          d->host_data.strides[1], p->level.val, p->base_log.val,
          p->input_lwe_dim.val, p->output_lwe_dim.val, p->sk_index.val,
          p->ctx.val);
      Dependence *dep =
          new Dependence(loc, out, nullptr, true, true, d->chunk_id);
      return dep;
    } else {
      // Schedule the keyswitch kernel on the GPU
      cudaStream_t *s = (cudaStream_t *)p->dfg->get_gpu_stream(loc);
      void *ct0_gpu = d->device_data;
      void *out_gpu = cuda_malloc_async(data_size, s, loc);
      void *ksk_gpu = p->ctx.val->get_ksk_gpu(
          p->level.val, p->input_lwe_dim.val, p->output_lwe_dim.val, loc, s,
          p->sk_index.val);
      cuda_keyswitch_lwe_ciphertext_vector_64(
          s, loc, out_gpu, ct0_gpu, ksk_gpu, p->input_lwe_dim.val,
          p->output_lwe_dim.val, p->base_log.val, p->level.val, num_samples);
      Dependence *dep =
          new Dependence(loc, out, out_gpu, false, false, d->chunk_id);
      return dep;
    }
  };
  Dependence *idep = p->input_streams[0]->get(loc, chunk_id);
  if (p->output_streams[0]->need_new_gen(chunk_id))
    p->output_streams[0]->put(sched(idep), chunk_id);
}

void memref_bootstrap_lwe_u64_process(Process *p, int32_t loc, int32_t chunk_id,
                                      uint64_t *out_ptr) {
  assert(p->output_size.val == p->glwe_dim.val * p->poly_size.val + 1);

  Dependence *idep1 = p->input_streams[1]->get(host_location, chunk_id);
  MemRef2 &mtlu = idep1->host_data;
  uint32_t num_lut_vectors = mtlu.sizes[0];
  uint64_t glwe_ct_len =
      p->poly_size.val * (p->glwe_dim.val + 1) * num_lut_vectors;
  uint64_t glwe_ct_size = glwe_ct_len * sizeof(uint64_t);
  uint64_t *glwe_ct = (uint64_t *)malloc(glwe_ct_size);
  auto tlu = mtlu.aligned + mtlu.offset;
  // Glwe trivial encryption
  size_t pos = 0, postlu = 0;
  for (size_t l = 0; l < num_lut_vectors; ++l) {
    for (size_t i = 0; i < p->poly_size.val * p->glwe_dim.val; i++) {
      glwe_ct[pos++] = 0;
    }
    for (size_t i = 0; i < p->poly_size.val; i++) {
      glwe_ct[pos++] = tlu[postlu++];
    }
  }

  auto sched = [&](Dependence *d0, Dependence *d1, uint64_t *glwe_ct,
                   std::vector<size_t> &lut_indexes, cudaStream_t *s,
                   int32_t loc) {
    uint64_t num_samples = d0->host_data.sizes[0];
    MemRef2 out = {
        0, 0, 0, {num_samples, p->output_size.val}, {p->output_size.val, 1}};
    size_t data_size = memref_get_data_size(out);

    // Move test vector indexes to the GPU, the test vector indexes is set of 0
    uint32_t lwe_idx = 0,
             test_vector_idxes_size = num_samples * sizeof(uint64_t);
    uint64_t *test_vector_idxes = (uint64_t *)malloc(test_vector_idxes_size);
    if (lut_indexes.size() == 1) {
      memset((void *)test_vector_idxes, lut_indexes[0], test_vector_idxes_size);
    } else {
      assert(lut_indexes.size() == num_samples);
      for (size_t i = 0; i < num_samples; ++i)
        test_vector_idxes[i] = lut_indexes[i];
    }
    if (loc == host_location) {
      // If it is not profitable to offload, schedule kernel on CPU
      out.allocated = out.aligned =
          (uint64_t *)((out_ptr != nullptr) ? out_ptr : malloc(data_size));
      if (lut_indexes.size() == 1)
        memref_batched_bootstrap_lwe_u64(
            out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
            out.strides[0], out.strides[1], d0->host_data.allocated,
            d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
            d0->host_data.sizes[1], d0->host_data.strides[0],
            d0->host_data.strides[1], d1->host_data.allocated,
            d1->host_data.aligned, d1->host_data.offset, d1->host_data.sizes[1],
            d1->host_data.strides[1], p->input_lwe_dim.val, p->poly_size.val,
            p->level.val, p->base_log.val, p->glwe_dim.val, p->sk_index.val,
            p->ctx.val);
      else
        memref_batched_mapped_bootstrap_lwe_u64(
            out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
            out.strides[0], out.strides[1], d0->host_data.allocated,
            d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
            d0->host_data.sizes[1], d0->host_data.strides[0],
            d0->host_data.strides[1], d1->host_data.allocated,
            d1->host_data.aligned, d1->host_data.offset, d1->host_data.sizes[0],
            d1->host_data.sizes[1], d1->host_data.strides[0],
            d1->host_data.strides[1], p->input_lwe_dim.val, p->poly_size.val,
            p->level.val, p->base_log.val, p->glwe_dim.val, p->sk_index.val,
            p->ctx.val);
      Dependence *dep =
          new Dependence(loc, out, nullptr, true, true, d0->chunk_id);
      free(glwe_ct);
      return dep;
    } else {
      // Schedule the bootstrap kernel on the GPU
      void *glwe_ct_gpu = cuda_malloc_async(glwe_ct_size, s, loc);
      cuda_memcpy_async_to_gpu(glwe_ct_gpu, glwe_ct, glwe_ct_size, s, loc);
      void *test_vector_idxes_gpu =
          cuda_malloc_async(test_vector_idxes_size, s, loc);
      cuda_memcpy_async_to_gpu(test_vector_idxes_gpu, (void *)test_vector_idxes,
                               test_vector_idxes_size, s, loc);
      int8_t *pbs_buffer = p->dfg->gpus[loc].get_pbs_buffer(
          p->glwe_dim.val, p->poly_size.val, num_samples);
      void *ct0_gpu = d0->device_data;
      void *out_gpu = cuda_malloc_async(data_size, s, loc);
      void *fbsk_gpu = p->ctx.val->get_bsk_gpu(
          p->input_lwe_dim.val, p->poly_size.val, p->level.val, p->glwe_dim.val,
          loc, s, p->sk_index.val);
      cuda_bootstrap_amortized_lwe_ciphertext_vector_64(
          s, loc, out_gpu, glwe_ct_gpu, test_vector_idxes_gpu, ct0_gpu,
          fbsk_gpu, (int8_t *)pbs_buffer, p->input_lwe_dim.val, p->glwe_dim.val,
          p->poly_size.val, p->base_log.val, p->level.val, num_samples,
          lut_indexes.size(), lwe_idx, cuda_get_max_shared_memory(loc));
      cuda_drop_async(test_vector_idxes_gpu, s, loc);
      cuda_drop_async(glwe_ct_gpu, s, loc);
      Dependence *dep =
          new Dependence(loc, out, out_gpu, false, false, d0->chunk_id);
      // As streams are not synchronized, we can only free this vector
      // after a later synchronization point where we are guaranteed that
      // this vector is no longer needed.
      p->dfg->register_stream_order_dependent_allocation(test_vector_idxes);
      p->dfg->register_stream_order_dependent_allocation(glwe_ct);
      return dep;
    }
  };

  // If this is a mapped TLU
  // TODO: for now we do not provide more advanced ways of selecting
  bool mapped = (p->input_streams[1]->dep->host_data.sizes[0] > 1);
  std::vector<size_t> lut_indexes;
  if (mapped) {
    lut_indexes.resize(num_lut_vectors);
    std::iota(lut_indexes.begin(), lut_indexes.end(), 0);
  } else {
    lut_indexes.push_back(0);
  }

  cudaStream_t *cstream = (cudaStream_t *)p->dfg->get_gpu_stream(loc);
  Dependence *idep0 = p->input_streams[0]->get(loc, chunk_id);
  if (p->output_streams[0]->need_new_gen(chunk_id))
    p->output_streams[0]->put(
        sched(idep0, idep1, glwe_ct, lut_indexes, cstream, loc), chunk_id);
}

void memref_add_lwe_ciphertexts_u64_process(Process *p, int32_t loc,
                                            int32_t chunk_id,
                                            uint64_t *out_ptr) {
  auto sched = [&](Dependence *d0, Dependence *d1, cudaStream_t *s,
                   int32_t loc) {
    assert(d0->host_data.sizes[0] == d1->host_data.sizes[0]);
    assert(d0->host_data.sizes[1] == d1->host_data.sizes[1]);
    assert(d0->chunk_id == d1->chunk_id);
    uint64_t num_samples = d0->host_data.sizes[0];
    MemRef2 out = {0,
                   0,
                   0,
                   {num_samples, d0->host_data.sizes[1]},
                   {d0->host_data.sizes[1], 1}};
    size_t data_size = memref_get_data_size(out);
    if (loc == host_location) {
      // If it is not profitable to offload, schedule kernel on CPU
      out.allocated = out.aligned =
          (uint64_t *)((out_ptr != nullptr) ? out_ptr : malloc(data_size));
      memref_batched_add_lwe_ciphertexts_u64(
          out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
          out.strides[0], out.strides[1], d0->host_data.allocated,
          d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
          d0->host_data.sizes[1], d0->host_data.strides[0],
          d0->host_data.strides[1], d1->host_data.allocated,
          d1->host_data.aligned, d1->host_data.offset, d1->host_data.sizes[0],
          d1->host_data.sizes[1], d1->host_data.strides[0],
          d1->host_data.strides[1]);
      Dependence *dep =
          new Dependence(loc, out, nullptr, true, true, d0->chunk_id);
      return dep;
    } else {
      // Schedule the kernel on the GPU
      void *out_gpu = cuda_malloc_async(data_size, s, loc);
      cuda_add_lwe_ciphertext_vector_64(
          s, loc, out_gpu, d0->device_data, d1->device_data,
          d0->host_data.sizes[1] - 1, num_samples);
      Dependence *dep =
          new Dependence(loc, out, out_gpu, false, false, d0->chunk_id);
      return dep;
    }
  };
  Dependence *idep0 = p->input_streams[0]->get(loc, chunk_id);
  Dependence *idep1 = p->input_streams[1]->get(loc, chunk_id);
  if (p->output_streams[0]->need_new_gen(chunk_id))
    p->output_streams[0]->put(
        sched(idep0, idep1, (cudaStream_t *)p->dfg->get_gpu_stream(loc), loc),
        chunk_id);
}

void memref_add_plaintext_lwe_ciphertext_u64_process(Process *p, int32_t loc,
                                                     int32_t chunk_id,
                                                     uint64_t *out_ptr) {
  auto sched = [&](Dependence *d0, Dependence *d1, cudaStream_t *s,
                   int32_t loc) {
    assert(d0->host_data.sizes[0] == d1->host_data.sizes[1] ||
           d1->host_data.sizes[1] == 1);
    assert(d0->chunk_id == d1->chunk_id);
    uint64_t num_samples = d0->host_data.sizes[0];
    MemRef2 out = {0,
                   0,
                   0,
                   {num_samples, d0->host_data.sizes[1]},
                   {d0->host_data.sizes[1], 1}};
    size_t data_size = memref_get_data_size(out);
    if (loc == host_location) {
      // If it is not profitable to offload, schedule kernel on CPU
      out.allocated = out.aligned =
          (uint64_t *)((out_ptr != nullptr) ? out_ptr : malloc(data_size));
      if (d1->host_data.sizes[1] == 1) // Constant case - or single sample
        memref_batched_add_plaintext_cst_lwe_ciphertext_u64(
            out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
            out.strides[0], out.strides[1], d0->host_data.allocated,
            d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
            d0->host_data.sizes[1], d0->host_data.strides[0],
            d0->host_data.strides[1],
            *(d1->host_data.aligned + d1->host_data.offset));
      else
        memref_batched_add_plaintext_lwe_ciphertext_u64(
            out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
            out.strides[0], out.strides[1], d0->host_data.allocated,
            d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
            d0->host_data.sizes[1], d0->host_data.strides[0],
            d0->host_data.strides[1], d1->host_data.allocated,
            d1->host_data.aligned, d1->host_data.offset, d1->host_data.sizes[1],
            d1->host_data.strides[1]);
      Dependence *dep =
          new Dependence(loc, out, nullptr, true, true, d0->chunk_id);
      return dep;
    } else {
      // Schedule the kernel on the GPU
      void *out_gpu = cuda_malloc_async(data_size, s, loc);
      cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
          s, loc, out_gpu, d0->device_data, d1->device_data,
          d0->host_data.sizes[1] - 1, num_samples);
      Dependence *dep =
          new Dependence(loc, out, out_gpu, false, false, d0->chunk_id);
      return dep;
    }
  };
  Dependence *idep0 = p->input_streams[0]->get(loc, chunk_id);
  Dependence *idep1 = p->input_streams[1]->get(loc, chunk_id);
  if (p->output_streams[0]->need_new_gen(chunk_id))
    p->output_streams[0]->put(
        sched(idep0, idep1, (cudaStream_t *)p->dfg->get_gpu_stream(loc), loc),
        chunk_id);
}

void memref_mul_cleartext_lwe_ciphertext_u64_process(Process *p, int32_t loc,
                                                     int32_t chunk_id,
                                                     uint64_t *out_ptr) {
  auto sched = [&](Dependence *d0, Dependence *d1, cudaStream_t *s,
                   int32_t loc) {
    assert(d0->host_data.sizes[0] == d1->host_data.sizes[1] ||
           d1->host_data.sizes[1] == 1);
    assert(d0->chunk_id == d1->chunk_id);
    uint64_t num_samples = d0->host_data.sizes[0];
    MemRef2 out = {0,
                   0,
                   0,
                   {num_samples, d0->host_data.sizes[1]},
                   {d0->host_data.sizes[1], 1}};
    size_t data_size = memref_get_data_size(out);
    if (loc == host_location) {
      // If it is not profitable to offload, schedule kernel on CPU
      out.allocated = out.aligned =
          (uint64_t *)((out_ptr != nullptr) ? out_ptr : malloc(data_size));
      if (d1->host_data.sizes[1] == 1) // Constant case
        memref_batched_mul_cleartext_cst_lwe_ciphertext_u64(
            out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
            out.strides[0], out.strides[1], d0->host_data.allocated,
            d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
            d0->host_data.sizes[1], d0->host_data.strides[0],
            d0->host_data.strides[1],
            *(d1->host_data.aligned + d1->host_data.offset));
      else
        memref_batched_mul_cleartext_lwe_ciphertext_u64(
            out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
            out.strides[0], out.strides[1], d0->host_data.allocated,
            d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
            d0->host_data.sizes[1], d0->host_data.strides[0],
            d0->host_data.strides[1], d1->host_data.allocated,
            d1->host_data.aligned, d1->host_data.offset, d1->host_data.sizes[1],
            d1->host_data.strides[1]);
      Dependence *dep =
          new Dependence(loc, out, nullptr, true, true, d0->chunk_id);
      return dep;
    } else {
      // Schedule the keyswitch kernel on the GPU
      void *out_gpu = cuda_malloc_async(data_size, s, loc);
      cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
          s, loc, out_gpu, d0->device_data, d1->device_data,
          d0->host_data.sizes[1] - 1, num_samples);
      Dependence *dep =
          new Dependence(loc, out, out_gpu, false, false, d0->chunk_id);
      return dep;
    }
  };
  Dependence *idep0 = p->input_streams[0]->get(loc, chunk_id);
  Dependence *idep1 = p->input_streams[1]->get(loc, chunk_id);
  if (p->output_streams[0]->need_new_gen(chunk_id))
    p->output_streams[0]->put(
        sched(idep0, idep1, (cudaStream_t *)p->dfg->get_gpu_stream(loc), loc),
        chunk_id);
}

void memref_negate_lwe_ciphertext_u64_process(Process *p, int32_t loc,
                                              int32_t chunk_id,
                                              uint64_t *out_ptr) {
  auto sched = [&](Dependence *d0, cudaStream_t *s, int32_t loc) {
    uint64_t num_samples = d0->host_data.sizes[0];
    MemRef2 out = {0,
                   0,
                   0,
                   {num_samples, d0->host_data.sizes[1]},
                   {d0->host_data.sizes[1], 1}};
    size_t data_size = memref_get_data_size(out);
    if (loc == host_location) {
      // If it is not profitable to offload, schedule kernel on CPU
      out.allocated = out.aligned =
          (uint64_t *)((out_ptr != nullptr) ? out_ptr : malloc(data_size));
      memref_batched_negate_lwe_ciphertext_u64(
          out.allocated, out.aligned, out.offset, out.sizes[0], out.sizes[1],
          out.strides[0], out.strides[1], d0->host_data.allocated,
          d0->host_data.aligned, d0->host_data.offset, d0->host_data.sizes[0],
          d0->host_data.sizes[1], d0->host_data.strides[0],
          d0->host_data.strides[1]);
      Dependence *dep =
          new Dependence(loc, out, nullptr, true, true, d0->chunk_id);
      return dep;
    } else {
      // Schedule the kernel on the GPU
      void *out_gpu = cuda_malloc_async(data_size, s, loc);
      cuda_negate_lwe_ciphertext_vector_64(s, loc, out_gpu, d0->device_data,
                                           d0->host_data.sizes[1] - 1,
                                           num_samples);
      Dependence *dep =
          new Dependence(loc, out, out_gpu, false, false, d0->chunk_id);
      return dep;
    }
  };
  Dependence *idep0 = p->input_streams[0]->get(loc, chunk_id);
  if (p->output_streams[0]->need_new_gen(chunk_id))
    p->output_streams[0]->put(
        sched(idep0, (cudaStream_t *)p->dfg->get_gpu_stream(loc), loc),
        chunk_id);
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
    uint32_t ksk_index, void *context) {
  Process *p =
      make_process_1_1(dfg, sin1, sout, memref_keyswitch_lwe_u64_process);
  p->level.val = level;
  p->base_log.val = base_log;
  p->input_lwe_dim.val = input_lwe_dim;
  p->output_lwe_dim.val = output_lwe_dim;
  p->sk_index.val = ksk_index;
  p->output_size.val = output_size;
  p->ctx.val = (RuntimeContext *)context;
  static int count = 0;
  sprintf(p->name, "keyswitch_%d", count++);
}

void stream_emulator_make_memref_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context) {
  // The TLU does not need to be sent to GPU
  ((Stream *)sin2)->type = TS_STREAM_TYPE_X86_TO_X86_LSAP;
  Process *p =
      make_process_2_1(dfg, sin1, sin2, sout, memref_bootstrap_lwe_u64_process);
  p->input_lwe_dim.val = input_lwe_dim;
  p->poly_size.val = poly_size;
  p->level.val = level;
  p->base_log.val = base_log;
  p->glwe_dim.val = glwe_dim;
  p->sk_index.val = bsk_index;
  p->output_size.val = output_size;
  p->ctx.val = (RuntimeContext *)context;
  static int count = 0;
  sprintf(p->name, "bootstrap_%d", count++);
}

void stream_emulator_make_memref_batched_add_lwe_ciphertexts_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->batched_stream = true;
  ((Stream *)sin2)->ct_stream = true;
  stream_emulator_make_memref_add_lwe_ciphertexts_u64_process(dfg, sin1, sin2,
                                                              sout);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void stream_emulator_make_memref_batched_add_plaintext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->batched_stream = true;
  ((Stream *)sin2)->pt_stream = true;
  stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process(
      dfg, sin1, sin2, sout);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}
void stream_emulator_make_memref_batched_add_plaintext_cst_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->const_stream = true;
  ((Stream *)sin2)->pt_stream = true;
  stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process(
      dfg, sin1, sin2, sout);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void stream_emulator_make_memref_batched_mul_cleartext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->batched_stream = true;
  ((Stream *)sin2)->pt_stream = true;
  stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process(
      dfg, sin1, sin2, sout);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}
void stream_emulator_make_memref_batched_mul_cleartext_cst_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->const_stream = true;
  ((Stream *)sin2)->pt_stream = true;
  stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process(
      dfg, sin1, sin2, sout);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void stream_emulator_make_memref_batched_negate_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sout) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  stream_emulator_make_memref_negate_lwe_ciphertext_u64_process(dfg, sin1,
                                                                sout);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void stream_emulator_make_memref_batched_keyswitch_lwe_u64_process(
    void *dfg, void *sin1, void *sout, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t output_size,
    uint32_t ksk_index, void *context) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  stream_emulator_make_memref_keyswitch_lwe_u64_process(
      dfg, sin1, sout, level, base_log, input_lwe_dim, output_lwe_dim,
      output_size, ksk_index, context);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void stream_emulator_make_memref_batched_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->const_stream = true;
  ((Stream *)sin2)->ct_stream = true;
  stream_emulator_make_memref_bootstrap_lwe_u64_process(
      dfg, sin1, sin2, sout, input_lwe_dim, poly_size, level, base_log,
      glwe_dim, output_size, bsk_index, context);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void stream_emulator_make_memref_batched_mapped_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context) {
  ((Stream *)sin1)->batched_stream = true;
  ((Stream *)sin1)->ct_stream = true;
  ((Stream *)sin2)->batched_stream = true;
  ((Stream *)sin2)->ct_stream = true;
  stream_emulator_make_memref_bootstrap_lwe_u64_process(
      dfg, sin1, sin2, sout, input_lwe_dim, poly_size, level, base_log,
      glwe_dim, output_size, bsk_index, context);
  ((Stream *)sout)->batched_stream = true;
  ((Stream *)sout)->ct_stream = true;
}

void *stream_emulator_make_uint64_stream(const char *name, stream_type stype) {
  return (void *)new Stream(stype, name);
}
void stream_emulator_put_uint64(void *stream, uint64_t e) {
  Stream *s = (Stream *)stream;
  uint64_t *data = (uint64_t *)malloc(sizeof(uint64_t));
  *data = e;
  MemRef2 m = {data, data, 0, {1, 1}, {1, 1}};
  Dependence *dep = new Dependence(host_location, m, nullptr, true, true);
  s->put(dep);
  s->generation++;
}
uint64_t stream_emulator_get_uint64(void *stream) {
  uint64_t res;
  auto s = (Stream *)stream;
  MemRef2 m = {&res, &res, 0, {1, 1}, {1, 1}};
  s->get_on_host(m);
  return res;
}

void *stream_emulator_make_memref_stream(const char *name, stream_type stype) {
  return (void *)new Stream(stype, name);
}
void stream_emulator_put_memref(void *stream, uint64_t *allocated,
                                uint64_t *aligned, uint64_t offset,
                                uint64_t size, uint64_t stride) {
  assert(stride == 1 && "Strided memrefs not supported");
  Stream *s = (Stream *)stream;
  MemRef2 m = {allocated, aligned, offset, {1, size}, {size, stride}};
  Dependence *dep =
      new Dependence(host_location, memref_copy_alloc(m), nullptr, true, true);
  s->put(dep);
  s->generation++;
}
void stream_emulator_get_memref(void *stream, uint64_t *out_allocated,
                                uint64_t *out_aligned, uint64_t out_offset,
                                uint64_t out_size, uint64_t out_stride) {
  assert(out_stride == 1 && "Strided memrefs not supported");
  MemRef2 mref = {out_allocated,
                  out_aligned,
                  out_offset,
                  {1, out_size},
                  {out_size, out_stride}};
  auto s = (Stream *)stream;
  s->get_on_host(mref);
}

void *stream_emulator_make_memref_batch_stream(const char *name,
                                               stream_type stype) {
  return (void *)new Stream(stype, name);
}
void stream_emulator_put_memref_batch(void *stream, uint64_t *allocated,
                                      uint64_t *aligned, uint64_t offset,
                                      uint64_t size0, uint64_t size1,
                                      uint64_t stride0, uint64_t stride1) {
  assert(stride1 == 1 && "Strided memrefs not supported");
  Stream *s = (Stream *)stream;
  MemRef2 m = {allocated, aligned, offset, {size0, size1}, {stride0, stride1}};
  Dependence *dep =
      new Dependence(host_location, memref_copy_alloc(m), nullptr, true, true);
  s->put(dep);
  s->generation++;
}
void stream_emulator_get_memref_batch(void *stream, uint64_t *out_allocated,
                                      uint64_t *out_aligned,
                                      uint64_t out_offset, uint64_t out_size0,
                                      uint64_t out_size1, uint64_t out_stride0,
                                      uint64_t out_stride1) {
  assert(out_stride1 == 1 && "Strided memrefs not supported");
  MemRef2 mref = {out_allocated,
                  out_aligned,
                  out_offset,
                  {out_size0, out_size1},
                  {out_stride0, out_stride1}};
  auto s = (Stream *)stream;
  s->get_on_host(mref);
}

void *stream_emulator_init() {
  int num;
  assert(cudaGetDeviceCount(&num) == cudaSuccess);
  num_devices = num;
  assert(num_devices > 0 && "No GPUs available on system.");
  char *env = getenv("SDFG_NUM_GPUS");
  if (env != nullptr) {
    size_t requested_gpus = strtoul(env, NULL, 10);
    if (requested_gpus == 0)
      warnx("WARNING: no GPUs requested (%lu available) - "
            "continuing with one (%lu) device.",
            num_devices, requested_gpus = 1);
    if (requested_gpus > num_devices)
      warnx("WARNING: requested more GPUs (%lu) than available (%lu) - "
            "continuing with available devices.",
            requested_gpus, num_devices);
    else
      num_devices = requested_gpus;
  }

  env = getenv("SDFG_DEVICE_TO_CORE_RATIO");
  if (env != nullptr)
    device_compute_factor = strtoul(env, NULL, 10);

  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_set_all_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter(topology, HWLOC_OBJ_CORE,
                                 HWLOC_TYPE_FILTER_KEEP_ALL);
  hwloc_topology_load(topology);
  num_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
  env = getenv("SDFG_NUM_THREADS");
  if (env != nullptr && strtoul(env, NULL, 10) != 0)
    num_cores = strtoul(env, NULL, 10);
  if (num_cores < 1)
    num_cores = 1;

  int device = next_device.fetch_add(1) % num_devices;
  return new GPU_DFG(device);
}
void stream_emulator_run(void *dfg) {}
void stream_emulator_delete(void *dfg) { delete (GPU_DFG *)dfg; }
#endif
