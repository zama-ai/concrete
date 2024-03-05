// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/stream_emulator_api.h"
#include "concretelang/Runtime/wrappers.h"
#include <cstdarg>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sched.h>
#include <thread>
#include <utility>
#include <vector>

namespace mlir {
namespace concretelang {
namespace stream_emulator {
namespace {

template <typename T> struct StreamBase {
  void put(T e) { q.push(e); }
  T get() {
    while (q.empty())
      sched_yield();
    T ret = q.front();
    q.pop();
    return ret;
  }
  bool empty() { return q.empty(); }

private:
  std::queue<T> q;
};
union Stream {
  StreamBase<uint64_t> *uint64_stream;
  StreamBase<MemRefDescriptor<1>> *memref_stream;

  Stream(StreamBase<uint64_t> *s) : uint64_stream(s) {}
  Stream(StreamBase<MemRefDescriptor<1>> *s) : memref_stream(s) {}
};

struct Void {};
union Param {
  Void _;
  uint32_t val;
};
union Context {
  Void _;
  mlir::concretelang::RuntimeContext *val;
};
struct Process {
  void terminate() { terminate_p = true; }
  bool terminate_p = false;
  std::vector<Stream> input_streams;
  std::vector<Stream> output_streams;
  Param level;
  Param base_log;
  Param input_lwe_dim;
  Param output_lwe_dim;
  Param poly_size;
  Param glwe_dim;
  Param precision;
  Param output_size;
  Param ksk_index;
  Param bsk_index;
  Context ctx;
  void (*fun)(Process *);
};

struct DFGraph {
  ~DFGraph() {
    for (auto p : dfg_processes)
      p->terminate();
  }
  void run() {
    for (auto p : dfg_processes) {
      std::thread process_thread(p->fun, p);
      process_thread.detach();
    }
  }
  std::vector<Process *> dfg_processes;
};

// Stream emulator processes
void memref_keyswitch_lwe_u64_process(Process *p) {
  while (!p->terminate_p) {
    MemRefDescriptor<1> ct0 = (p->input_streams[0]).memref_stream->get();
    MemRefDescriptor<1> out;
    out.sizes[0] = p->output_size.val;
    out.strides[0] = 1;
    out.offset = 0;
    out.allocated = out.aligned =
        (uint64_t *)malloc(out.sizes[0] * sizeof(uint64_t));
    memref_keyswitch_lwe_u64(
        out.allocated, out.aligned, out.offset, out.sizes[0], out.strides[0],
        ct0.allocated, ct0.aligned, ct0.offset, ct0.sizes[0], ct0.strides[0],
        p->level.val, p->base_log.val, p->input_lwe_dim.val,
        p->output_lwe_dim.val, p->ksk_index.val, p->ctx.val);
    (p->output_streams[0]).memref_stream->put(out);
  }
  delete p;
}

void memref_bootstrap_lwe_u64_process(Process *p) {
  while (!p->terminate_p) {
    MemRefDescriptor<1> ct0 = (p->input_streams[0]).memref_stream->get();
    MemRefDescriptor<1> tlu = (p->input_streams[1]).memref_stream->get();
    MemRefDescriptor<1> out;
    out.sizes[0] = p->output_size.val;
    out.strides[0] = 1;
    out.offset = 0;
    out.allocated = out.aligned =
        (uint64_t *)malloc(out.sizes[0] * sizeof(uint64_t));
    memref_bootstrap_lwe_u64(
        out.allocated, out.aligned, out.offset, out.sizes[0], out.strides[0],
        ct0.allocated, ct0.aligned, ct0.offset, ct0.sizes[0], ct0.strides[0],
        tlu.allocated, tlu.aligned, tlu.offset, tlu.sizes[0], tlu.strides[0],
        p->input_lwe_dim.val, p->poly_size.val, p->level.val, p->base_log.val,
        p->glwe_dim.val, p->bsk_index.val, p->ctx.val);
    (p->output_streams[0]).memref_stream->put(out);
  }
  delete p;
}

void memref_add_lwe_ciphertexts_u64_process(Process *p) {
  while (!p->terminate_p) {
    MemRefDescriptor<1> ct0 = (p->input_streams[0]).memref_stream->get();
    MemRefDescriptor<1> ct1 = (p->input_streams[1]).memref_stream->get();
    MemRefDescriptor<1> out = ct0;
    out.allocated = out.aligned =
        (uint64_t *)malloc(ct0.sizes[0] * sizeof(uint64_t));
    out.offset = 0;
    memref_add_lwe_ciphertexts_u64(
        out.allocated, out.aligned, out.offset, out.sizes[0], out.strides[0],
        ct0.allocated, ct0.aligned, ct0.offset, ct0.sizes[0], ct0.strides[0],
        ct1.allocated, ct1.aligned, ct1.offset, ct1.sizes[0], ct1.strides[0]);
    (p->output_streams[0]).memref_stream->put(out);
  }
  delete p;
}

void memref_add_plaintext_lwe_ciphertext_u64_process(Process *p) {
  while (!p->terminate_p) {
    MemRefDescriptor<1> ct0 = (p->input_streams[0]).memref_stream->get();
    uint64_t plaintext = (p->input_streams[1]).uint64_stream->get();
    MemRefDescriptor<1> out = ct0;
    out.allocated = out.aligned =
        (uint64_t *)malloc(ct0.sizes[0] * sizeof(uint64_t));
    out.offset = 0;
    memref_add_plaintext_lwe_ciphertext_u64(
        out.allocated, out.aligned, out.offset, out.sizes[0], out.strides[0],
        ct0.allocated, ct0.aligned, ct0.offset, ct0.sizes[0], ct0.strides[0],
        plaintext);
    (p->output_streams[0]).memref_stream->put(out);
  }
  delete p;
}

void memref_mul_cleartext_lwe_ciphertext_u64_process(Process *p) {
  while (!p->terminate_p) {
    MemRefDescriptor<1> ct0 = (p->input_streams[0]).memref_stream->get();
    uint64_t cleartext = (p->input_streams[1]).uint64_stream->get();
    MemRefDescriptor<1> out = ct0;
    out.allocated = out.aligned =
        (uint64_t *)malloc(ct0.sizes[0] * sizeof(uint64_t));
    out.offset = 0;
    memref_mul_cleartext_lwe_ciphertext_u64(
        out.allocated, out.aligned, out.offset, out.sizes[0], out.strides[0],
        ct0.allocated, ct0.aligned, ct0.offset, ct0.sizes[0], ct0.strides[0],
        cleartext);
    (p->output_streams[0]).memref_stream->put(out);
  }
  delete p;
}

void memref_negate_lwe_ciphertext_u64_process(Process *p) {
  while (!p->terminate_p) {
    MemRefDescriptor<1> ct0 = (p->input_streams[0]).memref_stream->get();
    MemRefDescriptor<1> out = ct0;
    out.allocated = out.aligned =
        (uint64_t *)malloc(ct0.sizes[0] * sizeof(uint64_t));
    out.offset = 0;
    memref_negate_lwe_ciphertext_u64(
        out.allocated, out.aligned, out.offset, out.sizes[0], out.strides[0],
        ct0.allocated, ct0.aligned, ct0.offset, ct0.sizes[0], ct0.strides[0]);
    (p->output_streams[0]).memref_stream->put(out);
  }
  delete p;
}

} // namespace
} // namespace stream_emulator
} // namespace concretelang
} // namespace mlir

// Code generation interface
void stream_emulator_make_memref_add_lwe_ciphertexts_u64_process(void *dfg,
                                                                 void *sin1,
                                                                 void *sin2,
                                                                 void *sout) {
  mlir::concretelang::stream_emulator::Process *p =
      new mlir::concretelang::stream_emulator::Process;
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin1);
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin2);
  p->output_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sout);
  p->fun = mlir::concretelang::stream_emulator::
      memref_add_lwe_ciphertexts_u64_process;
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)
      ->dfg_processes.push_back(p);
}

void stream_emulator_make_memref_add_plaintext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  mlir::concretelang::stream_emulator::Process *p =
      new mlir::concretelang::stream_emulator::Process;
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin1);
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<uint64_t> *)sin2);
  p->output_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sout);
  p->fun = mlir::concretelang::stream_emulator::
      memref_add_plaintext_lwe_ciphertext_u64_process;
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)
      ->dfg_processes.push_back(p);
}

void stream_emulator_make_memref_mul_cleartext_lwe_ciphertext_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout) {
  mlir::concretelang::stream_emulator::Process *p =
      new mlir::concretelang::stream_emulator::Process;
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin1);
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<uint64_t> *)sin2);
  p->output_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sout);
  p->fun = mlir::concretelang::stream_emulator::
      memref_mul_cleartext_lwe_ciphertext_u64_process;
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)
      ->dfg_processes.push_back(p);
}

void stream_emulator_make_memref_negate_lwe_ciphertext_u64_process(void *dfg,
                                                                   void *sin1,
                                                                   void *sout) {
  mlir::concretelang::stream_emulator::Process *p =
      new mlir::concretelang::stream_emulator::Process;
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin1);
  p->output_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sout);
  p->fun = mlir::concretelang::stream_emulator::
      memref_negate_lwe_ciphertext_u64_process;
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)
      ->dfg_processes.push_back(p);
}

void stream_emulator_make_memref_keyswitch_lwe_u64_process(
    void *dfg, void *sin1, void *sout, uint32_t level, uint32_t base_log,
    uint32_t input_lwe_dim, uint32_t output_lwe_dim, uint32_t output_size,
    uint32_t ksk_index, void *context) {
  mlir::concretelang::stream_emulator::Process *p =
      new mlir::concretelang::stream_emulator::Process;
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin1);
  p->output_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sout);
  p->level.val = level;
  p->base_log.val = base_log;
  p->input_lwe_dim.val = input_lwe_dim;
  p->output_lwe_dim.val = output_lwe_dim;
  p->output_size.val = output_size;
  p->ksk_index.val = ksk_index;
  p->ctx.val = (mlir::concretelang::RuntimeContext *)context;
  p->fun =
      mlir::concretelang::stream_emulator::memref_keyswitch_lwe_u64_process;
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)
      ->dfg_processes.push_back(p);
}

void stream_emulator_make_memref_bootstrap_lwe_u64_process(
    void *dfg, void *sin1, void *sin2, void *sout, uint32_t input_lwe_dim,
    uint32_t poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim,
    uint32_t output_size, uint32_t bsk_index, void *context) {
  mlir::concretelang::stream_emulator::Process *p =
      new mlir::concretelang::stream_emulator::Process;
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin1);
  p->input_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sin2);
  p->output_streams.push_back(
      (mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
          sout);
  p->input_lwe_dim.val = input_lwe_dim;
  p->poly_size.val = poly_size;
  p->level.val = level;
  p->base_log.val = base_log;
  p->glwe_dim.val = glwe_dim;
  p->output_size.val = output_size;
  p->bsk_index.val = bsk_index;
  p->ctx.val = (mlir::concretelang::RuntimeContext *)context;
  p->fun =
      mlir::concretelang::stream_emulator::memref_bootstrap_lwe_u64_process;
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)
      ->dfg_processes.push_back(p);
}

void *stream_emulator_make_uint64_stream(const char *name, stream_type stype) {
  return (void *)new mlir::concretelang::stream_emulator::StreamBase<uint64_t>;
}
void stream_emulator_put_uint64(void *stream, uint64_t e) {
  ((mlir::concretelang::stream_emulator::StreamBase<uint64_t> *)stream)->put(e);
}
uint64_t stream_emulator_get_uint64(void *stream) {
  return ((mlir::concretelang::stream_emulator::StreamBase<uint64_t> *)stream)
      ->get();
}

void *stream_emulator_make_memref_stream(const char *name, stream_type stype) {
  return (void *)new mlir::concretelang::stream_emulator::StreamBase<
      MemRefDescriptor<1>>;
}
void stream_emulator_put_memref(void *stream, uint64_t *allocated,
                                uint64_t *aligned, uint64_t offset,
                                uint64_t size, uint64_t stride) {
  ((mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
       stream)
      ->put({allocated, aligned, offset, {size}, {stride}});
}
void stream_emulator_get_memref(void *stream, uint64_t *out_allocated,
                                uint64_t *out_aligned, uint64_t out_offset,
                                uint64_t out_size, uint64_t out_stride) {
  MemRefDescriptor<1> mref =
      ((mlir::concretelang::stream_emulator::StreamBase<MemRefDescriptor<1>> *)
           stream)
          ->get();
  memref_copy_one_rank(mref.allocated, mref.aligned, mref.offset, mref.sizes[0],
                       mref.strides[0], out_allocated, out_aligned, out_offset,
                       out_size, out_stride);
  free(mref.allocated);
}

void *stream_emulator_make_memref_batch_stream(const char *name,
                                               stream_type stype) {
  assert(0 && "Batched operations not implemented in the StreamEmulator.");
}
void stream_emulator_put_memref_batch(void *stream, uint64_t *allocated,
                                      uint64_t *aligned, uint64_t offset,
                                      uint64_t size0, uint64_t size1,
                                      uint64_t stride0, uint64_t stride1) {
  assert(0 && "Batched operations not implemented in the StreamEmulator.");
}
void stream_emulator_get_memref_batch(void *stream, uint64_t *out_allocated,
                                      uint64_t *out_aligned,
                                      uint64_t out_offset, uint64_t out_size0,
                                      uint64_t out_size1, uint64_t out_stride0,
                                      uint64_t out_stride1) {
  assert(0 && "Batched operations not implemented in the StreamEmulator.");
}

void *stream_emulator_init() {
#ifdef CORNAMI_AVAILABLE
  // TODO: check/update against new info on Cornami API
  fhestream *pfhestream = new fhestream();
  pfhestream->initTopology();
  return (void *)pfhestream;
#else
  return (void *)new mlir::concretelang::stream_emulator::DFGraph;
#endif
}

void stream_emulator_run(void *dfg) {
#ifdef CORNAMI_AVAILABLE
  ((fhestream *)dfg)->FinalizeAndRun();
#else
  ((mlir::concretelang::stream_emulator::DFGraph *)dfg)->run();
#endif
}

void stream_emulator_delete(void *dfg) {
#ifdef CORNAMI_AVAILABLE
  delete ((fhestream *)dfg);
#else
  delete ((mlir::concretelang::stream_emulator::DFGraph *)dfg);
#endif
}
