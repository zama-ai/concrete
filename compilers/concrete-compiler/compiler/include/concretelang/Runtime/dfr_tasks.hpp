// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_TASKS_HPP
#define CONCRETELANG_DFR_TASKS_HPP
#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED

namespace mlir {
namespace concretelang {
namespace dfr {

using namespace hpx;
typedef struct dfr_refcounted_future {
  hpx::shared_future<void *> *future;
  std::atomic<std::size_t> count;
  bool cloned_memref_p;
  dfr_refcounted_future(hpx::shared_future<void *> *f, size_t c, bool clone_p)
      : future(f), count(c), cloned_memref_p(clone_p) {}
} dfr_refcounted_future_t, *dfr_refcounted_future_p;

// Determine where new task should run.  For now just round-robin
// distribution - TODO: optimise.
static inline size_t dfr_get_next_execution_locality() {
  static std::atomic<std::size_t> next_locality{1};

  size_t next_loc = next_locality.fetch_add(1);

  return next_loc % num_nodes;
}

void dfr_create_async_task_impl(wfnptr wfn, void *ctx,
                                std::vector<void *> &refcounted_futures,
                                std::vector<size_t> &param_sizes,
                                std::vector<uint64_t> &param_types,
                                std::vector<void *> &outputs,
                                std::vector<size_t> &output_sizes,
                                std::vector<uint64_t> &output_types) {
  // Take a reference on each future argument
  for (auto rcf : refcounted_futures)
    ((dfr_refcounted_future_p)rcf)->count.fetch_add(1);

  // We pass functions by name - which is not strictly necessary in
  // shared memory as pointers suffice, but is needed in the
  // distributed case where the functions need to be located/loaded on
  // the node.
  auto wfnname =
      _dfr_node_level_work_function_registry->getWorkFunctionName((void *)wfn);
  hpx::future<hpx::future<OpaqueOutputData>> oodf;

  // In order to allow complete dataflow semantics for
  // communication/synchronization, we split tasks in two parts: an
  // execution body that is scheduled once all input dependences are
  // satisfied, which generates a future on a tuple of outputs, which
  // is then further split into a tuple of futures and provide
  // individual synchronization for each return independently.
  GenericComputeClient *gcc_target = &gcc[dfr_get_next_execution_locality()];
  switch (refcounted_futures.size()) {

#include "concretelang/Runtime/generated/dfr_dataflow_inputs_cases.h"

  default:
    HPX_THROW_EXCEPTION(hpx::error::no_success, "_dfr_create_async_task",
                        "Error: number of task parameters not supported.");
  }

  switch (outputs.size()) {
  case 1:
    *((void **)outputs[0]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(hpx::dataflow(
            [refcounted_futures](
                hpx::future<OpaqueOutputData> oodf_in) -> void * {
              void *ret = oodf_in.get().outputs[0];
              for (auto rcf : refcounted_futures)
                _dfr_deallocate_future(rcf);
              return ret;
            },
            oodf)),
        1, output_types[0] == _DFR_TASK_ARG_MEMREF);
    break;

  case 2: {
    hpx::future<hpx::tuple<void *, void *>> &&ft = hpx::dataflow(
        [refcounted_futures](hpx::future<OpaqueOutputData> oodf_in)
            -> hpx::tuple<void *, void *> {
          std::vector<void *> outputs = std::move(oodf_in.get().outputs);
          for (auto rcf : refcounted_futures)
            _dfr_deallocate_future(rcf);
          return hpx::make_tuple<>(outputs[0], outputs[1]);
        },
        oodf);
    hpx::tuple<hpx::future<void *>, hpx::future<void *>> &&tf =
        hpx::split_future(std::move(ft));
    *((void **)outputs[0]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<0>(tf))), 1,
        output_types[0] == _DFR_TASK_ARG_MEMREF);
    *((void **)outputs[1]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<1>(tf))), 1,
        output_types[1] == _DFR_TASK_ARG_MEMREF);
    break;
  }

  case 3: {
    hpx::future<hpx::tuple<void *, void *, void *>> &&ft = hpx::dataflow(
        [refcounted_futures](hpx::future<OpaqueOutputData> oodf_in)
            -> hpx::tuple<void *, void *, void *> {
          std::vector<void *> outputs = std::move(oodf_in.get().outputs);
          for (auto rcf : refcounted_futures)
            _dfr_deallocate_future(rcf);
          return hpx::make_tuple<>(outputs[0], outputs[1], outputs[2]);
        },
        oodf);
    hpx::tuple<hpx::future<void *>, hpx::future<void *>, hpx::future<void *>>
        &&tf = hpx::split_future(std::move(ft));
    *((void **)outputs[0]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<0>(tf))), 1,
        output_types[0] == _DFR_TASK_ARG_MEMREF);
    *((void **)outputs[1]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<1>(tf))), 1,
        output_types[1] == _DFR_TASK_ARG_MEMREF);
    *((void **)outputs[2]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<2>(tf))), 1,
        output_types[2] == _DFR_TASK_ARG_MEMREF);
    break;
  }

  default:
    HPX_THROW_EXCEPTION(hpx::error::no_success, "_dfr_create_async_task",
                        "Error: number of task outputs not supported.");
  }
}

} // namespace dfr
} // namespace concretelang
} // namespace mlir

#endif
#endif
