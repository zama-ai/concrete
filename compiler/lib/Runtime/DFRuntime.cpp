// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

/// This file implements the dataflow runtime. It encapsulates all of
/// the underlying communication, parallelism, etc. and only exposes a
/// simplified interface for code generation in runtime_api.h
/// This hides the details of implementation, including of the HPX
/// framework currently used, from the code generation side.

#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED

#include <assert.h>
#include <hpx/barrier.hpp>
#include <hpx/future.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hwloc.h>
#include <omp.h>

#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/distributed_generic_task_server.hpp"
#include "concretelang/Runtime/runtime_api.h"
#include "concretelang/Runtime/time_util.h"

namespace mlir {
namespace concretelang {
namespace dfr {
namespace {
static std::vector<GenericComputeClient> gcc;
static hpx::lcos::barrier *_dfr_jit_phase_barrier;
static hpx::lcos::barrier *_dfr_startup_barrier;
static size_t num_nodes = 0;
static struct timespec init_timer, broadcast_timer, compute_timer, whole_timer;
} // namespace
} // namespace dfr
} // namespace concretelang
} // namespace mlir

using namespace hpx;

typedef struct dfr_refcounted_future {
  hpx::shared_future<void *> *future;
  std::atomic<std::size_t> count;
  bool cloned_memref_p;
  dfr_refcounted_future(hpx::shared_future<void *> *f, size_t c, bool clone_p)
      : future(f), count(c), cloned_memref_p(clone_p) {}
} dfr_refcounted_future_t, *dfr_refcounted_future_p;

// Ready futures are only used as inputs to tasks (never passed to
// await_future), so we only need to track the references in task
// creation.
void *_dfr_make_ready_future(void *in, size_t memref_clone_p) {
  return (void *)new dfr_refcounted_future_t(
      new hpx::shared_future<void *>(hpx::make_ready_future(in)), 1,
      memref_clone_p);
}

void *_dfr_await_future(void *in) {
  return static_cast<dfr_refcounted_future_p>(in)->future->get();
}

void _dfr_deallocate_future(void *in) {
  auto drf = static_cast<dfr_refcounted_future_p>(in);
  size_t prev_count = drf->count.fetch_sub(1);
  if (prev_count == 1) {
    // If this was a memref for which a clone was needed, deallocate first.
    if (drf->cloned_memref_p)
      free(
          (void *)(static_cast<StridedMemRefType<char, 1> *>(drf->future->get())
                       ->data));
    free(drf->future->get());
    delete (drf->future);
    delete drf;
  }
}

void _dfr_deallocate_future_data(void *in) {}

// Determine where new task should run.  For now just round-robin
// distribution - TODO: optimise.
static inline size_t _dfr_find_next_execution_locality() {
  static std::atomic<std::size_t> next_locality{1};

  size_t next_loc = next_locality.fetch_add(1);

  return next_loc % mlir::concretelang::dfr::num_nodes;
}

/// Runtime generic async_task.  Each first NUM_PARAMS pairs of
/// arguments in the variadic list corresponds to a void* pointer on a
/// hpx::future<void*> and the size of data within the future.  After
/// that come NUM_OUTPUTS pairs of hpx::future<void*>* and size_t for
/// the returns.
void _dfr_create_async_task(wfnptr wfn, void *ctx, size_t num_params,
                            size_t num_outputs, ...) {
  std::vector<void *> refcounted_futures;
  std::vector<size_t> param_sizes;
  std::vector<uint64_t> param_types;
  std::vector<void *> outputs;
  std::vector<size_t> output_sizes;
  std::vector<uint64_t> output_types;

  va_list args;
  va_start(args, num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    outputs.push_back(va_arg(args, void *));
    output_sizes.push_back(va_arg(args, uint64_t));
    output_types.push_back(va_arg(args, uint64_t));
  }
  for (size_t i = 0; i < num_params; ++i) {
    refcounted_futures.push_back(va_arg(args, void *));
    param_sizes.push_back(va_arg(args, uint64_t));
    param_types.push_back(va_arg(args, uint64_t));
  }
  va_end(args);

  // Take a reference on each future argument
  for (auto rcf : refcounted_futures)
    ((dfr_refcounted_future_p)rcf)->count.fetch_add(1);

  // We pass functions by name - which is not strictly necessary in
  // shared memory as pointers suffice, but is needed in the
  // distributed case where the functions need to be located/loaded on
  // the node.
  auto wfnname = mlir::concretelang::dfr::_dfr_node_level_work_function_registry
                     ->getWorkFunctionName((void *)wfn);
  hpx::future<hpx::future<mlir::concretelang::dfr::OpaqueOutputData>> oodf;

  // In order to allow complete dataflow semantics for
  // communication/synchronization, we split tasks in two parts: an
  // execution body that is scheduled once all input dependences are
  // satisfied, which generates a future on a tuple of outputs, which
  // is then further split into a tuple of futures and provide
  // individual synchronization for each return independently.
  mlir::concretelang::dfr::GenericComputeClient *gcc_target =
      &mlir::concretelang::dfr::gcc[_dfr_find_next_execution_locality()];
  switch (num_params) {
  case 0:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target,
         ctx]() -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        }));
    break;

  case 1:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {param0.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future));
    break;

  case 2:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future));
    break;

  case 3:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get(),
                                        param2.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future));
    break;

  case 4:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get(),
                                        param2.get(), param3.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future));
    break;

  case 5:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get(),
                                        param2.get(), param3.get(),
                                        param4.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future));
    break;

  case 6:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get(),
                                        param2.get(), param3.get(),
                                        param4.get(), param5.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future));
    break;

  case 7:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(), param3.get(),
              param4.get(), param5.get(), param6.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future));
    break;

  case 8:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(), param3.get(),
              param4.get(), param5.get(), param6.get(), param7.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future));
    break;

  case 9:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(),
              param3.get(), param4.get(), param5.get(),
              param6.get(), param7.get(), param8.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future));
    break;

  case 10:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(), param3.get(),
              param4.get(), param5.get(), param6.get(), param7.get(),
              param8.get(), param9.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future));
    break;

  case 11:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(), param3.get(),
              param4.get(), param5.get(), param6.get(), param7.get(),
              param8.get(), param9.get(), param10.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future));
    break;

  case 12:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(),  param3.get(),
              param4.get(), param5.get(), param6.get(),  param7.get(),
              param8.get(), param9.get(), param10.get(), param11.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future));
    break;

  case 13:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(), param1.get(), param2.get(),  param3.get(),
              param4.get(), param5.get(), param6.get(),  param7.get(),
              param8.get(), param9.get(), param10.get(), param11.get(),
              param12.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future));
    break;

  case 14:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(), param2.get(),  param3.get(),
              param4.get(),  param5.get(), param6.get(),  param7.get(),
              param8.get(),  param9.get(), param10.get(), param11.get(),
              param12.get(), param13.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future));
    break;

  case 15:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13,
                          hpx::shared_future<void *> param14)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(),  param2.get(),  param3.get(),
              param4.get(),  param5.get(),  param6.get(),  param7.get(),
              param8.get(),  param9.get(),  param10.get(), param11.get(),
              param12.get(), param13.get(), param14.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future,
        *((dfr_refcounted_future_p)refcounted_futures[14])->future));
    break;

  case 16:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13,
                          hpx::shared_future<void *> param14,
                          hpx::shared_future<void *> param15)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(),  param2.get(),  param3.get(),
              param4.get(),  param5.get(),  param6.get(),  param7.get(),
              param8.get(),  param9.get(),  param10.get(), param11.get(),
              param12.get(), param13.get(), param14.get(), param15.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future,
        *((dfr_refcounted_future_p)refcounted_futures[14])->future,
        *((dfr_refcounted_future_p)refcounted_futures[15])->future));
    break;

  case 17:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13,
                          hpx::shared_future<void *> param14,
                          hpx::shared_future<void *> param15,
                          hpx::shared_future<void *> param16)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(),  param2.get(),  param3.get(),
              param4.get(),  param5.get(),  param6.get(),  param7.get(),
              param8.get(),  param9.get(),  param10.get(), param11.get(),
              param12.get(), param13.get(), param14.get(), param15.get(),
              param16.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future,
        *((dfr_refcounted_future_p)refcounted_futures[14])->future,
        *((dfr_refcounted_future_p)refcounted_futures[15])->future,
        *((dfr_refcounted_future_p)refcounted_futures[16])->future));
    break;

  case 18:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13,
                          hpx::shared_future<void *> param14,
                          hpx::shared_future<void *> param15,
                          hpx::shared_future<void *> param16,
                          hpx::shared_future<void *> param17)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(),  param2.get(),  param3.get(),
              param4.get(),  param5.get(),  param6.get(),  param7.get(),
              param8.get(),  param9.get(),  param10.get(), param11.get(),
              param12.get(), param13.get(), param14.get(), param15.get(),
              param16.get(), param17.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future,
        *((dfr_refcounted_future_p)refcounted_futures[14])->future,
        *((dfr_refcounted_future_p)refcounted_futures[15])->future,
        *((dfr_refcounted_future_p)refcounted_futures[16])->future,
        *((dfr_refcounted_future_p)refcounted_futures[17])->future));
    break;

  case 19:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13,
                          hpx::shared_future<void *> param14,
                          hpx::shared_future<void *> param15,
                          hpx::shared_future<void *> param16,
                          hpx::shared_future<void *> param17,
                          hpx::shared_future<void *> param18)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(),  param2.get(),  param3.get(),
              param4.get(),  param5.get(),  param6.get(),  param7.get(),
              param8.get(),  param9.get(),  param10.get(), param11.get(),
              param12.get(), param13.get(), param14.get(), param15.get(),
              param16.get(), param17.get(), param18.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future,
        *((dfr_refcounted_future_p)refcounted_futures[14])->future,
        *((dfr_refcounted_future_p)refcounted_futures[15])->future,
        *((dfr_refcounted_future_p)refcounted_futures[16])->future,
        *((dfr_refcounted_future_p)refcounted_futures[17])->future,
        *((dfr_refcounted_future_p)refcounted_futures[18])->future));
    break;

  case 20:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, param_types, output_sizes, output_types,
         gcc_target, ctx](hpx::shared_future<void *> param0,
                          hpx::shared_future<void *> param1,
                          hpx::shared_future<void *> param2,
                          hpx::shared_future<void *> param3,
                          hpx::shared_future<void *> param4,
                          hpx::shared_future<void *> param5,
                          hpx::shared_future<void *> param6,
                          hpx::shared_future<void *> param7,
                          hpx::shared_future<void *> param8,
                          hpx::shared_future<void *> param9,
                          hpx::shared_future<void *> param10,
                          hpx::shared_future<void *> param11,
                          hpx::shared_future<void *> param12,
                          hpx::shared_future<void *> param13,
                          hpx::shared_future<void *> param14,
                          hpx::shared_future<void *> param15,
                          hpx::shared_future<void *> param16,
                          hpx::shared_future<void *> param17,
                          hpx::shared_future<void *> param18,
                          hpx::shared_future<void *> param19)
            -> hpx::future<mlir::concretelang::dfr::OpaqueOutputData> {
          std::vector<void *> params = {
              param0.get(),  param1.get(),  param2.get(),  param3.get(),
              param4.get(),  param5.get(),  param6.get(),  param7.get(),
              param8.get(),  param9.get(),  param10.get(), param11.get(),
              param12.get(), param13.get(), param14.get(), param15.get(),
              param16.get(), param17.get(), param18.get(), param19.get()};
          mlir::concretelang::dfr::OpaqueInputData oid(
              wfnname, params, param_sizes, param_types, output_sizes,
              output_types, ctx);
          return gcc_target->execute_task(oid);
        },
        *((dfr_refcounted_future_p)refcounted_futures[0])->future,
        *((dfr_refcounted_future_p)refcounted_futures[1])->future,
        *((dfr_refcounted_future_p)refcounted_futures[2])->future,
        *((dfr_refcounted_future_p)refcounted_futures[3])->future,
        *((dfr_refcounted_future_p)refcounted_futures[4])->future,
        *((dfr_refcounted_future_p)refcounted_futures[5])->future,
        *((dfr_refcounted_future_p)refcounted_futures[6])->future,
        *((dfr_refcounted_future_p)refcounted_futures[7])->future,
        *((dfr_refcounted_future_p)refcounted_futures[8])->future,
        *((dfr_refcounted_future_p)refcounted_futures[9])->future,
        *((dfr_refcounted_future_p)refcounted_futures[10])->future,
        *((dfr_refcounted_future_p)refcounted_futures[11])->future,
        *((dfr_refcounted_future_p)refcounted_futures[12])->future,
        *((dfr_refcounted_future_p)refcounted_futures[13])->future,
        *((dfr_refcounted_future_p)refcounted_futures[14])->future,
        *((dfr_refcounted_future_p)refcounted_futures[15])->future,
        *((dfr_refcounted_future_p)refcounted_futures[16])->future,
        *((dfr_refcounted_future_p)refcounted_futures[17])->future,
        *((dfr_refcounted_future_p)refcounted_futures[18])->future,
        *((dfr_refcounted_future_p)refcounted_futures[19])->future));
    break;

  default:
    HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_create_async_task",
                        "Error: number of task parameters not supported.");
  }

  switch (num_outputs) {
  case 1:
    *((void **)outputs[0]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(hpx::dataflow(
            [refcounted_futures](
                hpx::future<mlir::concretelang::dfr::OpaqueOutputData> oodf_in)
                -> void * {
              void *ret = oodf_in.get().outputs[0];
              for (auto rcf : refcounted_futures)
                _dfr_deallocate_future(rcf);
              return ret;
            },
            oodf)),
        1, output_types[0] == mlir::concretelang::dfr::_DFR_TASK_ARG_MEMREF);
    break;

  case 2: {
    hpx::future<hpx::tuple<void *, void *>> &&ft = hpx::dataflow(
        [refcounted_futures](
            hpx::future<mlir::concretelang::dfr::OpaqueOutputData> oodf_in)
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
        output_types[0] == mlir::concretelang::dfr::_DFR_TASK_ARG_MEMREF);
    *((void **)outputs[1]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<1>(tf))), 1,
        output_types[1] == mlir::concretelang::dfr::_DFR_TASK_ARG_MEMREF);
    break;
  }

  case 3: {
    hpx::future<hpx::tuple<void *, void *, void *>> &&ft = hpx::dataflow(
        [refcounted_futures](
            hpx::future<mlir::concretelang::dfr::OpaqueOutputData> oodf_in)
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
        output_types[0] == mlir::concretelang::dfr::_DFR_TASK_ARG_MEMREF);
    *((void **)outputs[1]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<1>(tf))), 1,
        output_types[1] == mlir::concretelang::dfr::_DFR_TASK_ARG_MEMREF);
    *((void **)outputs[2]) = (void *)new dfr_refcounted_future_t(
        new hpx::shared_future<void *>(std::move(hpx::get<2>(tf))), 1,
        output_types[2] == mlir::concretelang::dfr::_DFR_TASK_ARG_MEMREF);
    break;
  }

  default:
    HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_create_async_task",
                        "Error: number of task outputs not supported.");
  }
}

/***************************/
/* JIT execution support.  */
/***************************/
void _dfr_try_initialize();
namespace mlir {
namespace concretelang {
namespace dfr {
namespace {
static bool dfr_required_p = false;
static bool is_jit_p = false;
static bool is_root_node_p = true;
static bool use_omp_p = false;
} // namespace

void _dfr_set_required(bool is_required) {
  mlir::concretelang::dfr::dfr_required_p = is_required;
  if (mlir::concretelang::dfr::dfr_required_p) {
    _dfr_try_initialize();
  }
}
void _dfr_set_jit(bool is_jit) { mlir::concretelang::dfr::is_jit_p = is_jit; }
void _dfr_set_use_omp(bool use_omp) {
  mlir::concretelang::dfr::use_omp_p = use_omp;
}
bool _dfr_is_jit() { return mlir::concretelang::dfr::is_jit_p; }
bool _dfr_is_root_node() { return mlir::concretelang::dfr::is_root_node_p; }
bool _dfr_use_omp() { return mlir::concretelang::dfr::use_omp_p; }
bool _dfr_is_distributed() { return num_nodes > 1; }
} // namespace dfr
} // namespace concretelang
} // namespace mlir

void _dfr_register_work_function(wfnptr wfn) {
  mlir::concretelang::dfr::_dfr_node_level_work_function_registry
      ->getWorkFunctionName((void *)wfn);
}

/************************************/
/*  Initialization & Finalization.  */
/************************************/
namespace mlir {
namespace concretelang {
namespace dfr {
namespace {
static std::atomic<uint64_t> init_guard = {0};
static uint64_t uninitialised = 0;
static uint64_t active = 1;
static uint64_t terminated = 2;
} // namespace
} // namespace dfr
} // namespace concretelang
} // namespace mlir
static inline void _dfr_stop_impl() {
  if (mlir::concretelang::dfr::_dfr_is_root_node())
    hpx::apply([]() { hpx::finalize(); });
  hpx::stop();
  if (!mlir::concretelang::dfr::_dfr_is_root_node())
    exit(EXIT_SUCCESS);
}

static inline void _dfr_start_impl(int argc, char *argv[]) {
  BEGIN_TIME(&mlir::concretelang::dfr::init_timer);
  mlir::concretelang::dfr::dl_handle = dlopen(nullptr, RTLD_NOW);

  // If OpenMP is to be used, we need to force its initialization
  // before thread binding occurs. Otherwise OMP threads will be bound
  // to the core of the thread initializing the OMP runtime.
  if (mlir::concretelang::dfr::_dfr_use_omp()) {
#pragma omp parallel shared(mlir::concretelang::dfr::use_omp_p)
    {
#pragma omp critical
      mlir::concretelang::dfr::use_omp_p = true;
    }
  }

  if (argc == 0) {
    int nCores, nOMPThreads, nHPXThreads;
    std::string hpxThreadNum;

    std::vector<char *> parameters;
    parameters.push_back(const_cast<char *>("__dummy_dfr_HPX_program_name__"));
    parameters.push_back(const_cast<char *>("--hpx:print-bind"));

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_set_all_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_NONE);
    hwloc_topology_set_type_filter(topology, HWLOC_OBJ_CORE,
                                   HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topology);
    nCores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    if (nCores < 1)
      nCores = 1;

    // We do not directly handle this, but we should take into account
    // the choices made by the OpenMP runtime if we would be mixing
    // loop & dataflow parallelism.
    char *env = getenv("OMP_NUM_THREADS");
    if (mlir::concretelang::dfr::_dfr_use_omp() && env != nullptr)
      nOMPThreads = strtoul(env, NULL, 10);
    else if (mlir::concretelang::dfr::_dfr_use_omp())
      nOMPThreads = nCores;
    else
      nOMPThreads = 1;

    // Unless specified, we will consider that within each node loop
    // parallelism is the priority, so we would allocate either
    // ncores/OMP_NUM_THREADS or ncores-OMP_NUM_THREADS+1.  Both make
    // sense depending on whether we have very regular computation or
    // not - the latter being more conservative in that we will
    // exploit all cores, at the risk of oversubscribing.  Ideally the
    // distribution of hardware resources to the runtime systems
    // should be explicitly defined by the user.
    env = getenv("DFR_NUM_THREADS");
    if (env != nullptr) {
      nHPXThreads = strtoul(env, NULL, 10);
      parameters.push_back(const_cast<char *>("--hpx:threads"));
      hpxThreadNum = std::to_string(nHPXThreads);
      parameters.push_back(const_cast<char *>(hpxThreadNum.c_str()));
    } else
      nHPXThreads = nCores + 1 - nOMPThreads;
    if (nHPXThreads < 1)
      nHPXThreads = 1;

    // If the user does not provide their own config file, one is by
    // default located at the root of the concrete-compiler directory.
    env = getenv("HPX_CONFIG_FILE");
    // If no file is provided, try and check that the default is
    // available - otherwise use a basic default configuration.
#ifdef HPX_DEFAULT_CONFIG_FILE
    if (env == nullptr)
      if (access(HPX_DEFAULT_CONFIG_FILE, F_OK) == 0)
        env = const_cast<char *>(HPX_DEFAULT_CONFIG_FILE);
#endif
    if (env != nullptr) {
      parameters.push_back(const_cast<char *>("--hpx:config"));
      parameters.push_back(const_cast<char *>(env));
      hpx::start(nullptr, parameters.size(), parameters.data());
    } else {
      // Last resort configuration in case no config file could be
      // identified, provide some default values that make (some)
      // sense for homomorphic computations (stacks need to reflect
      // the size of ciphertexts rather than simple cleartext
      // scalars).
      if (std::find(parameters.begin(), parameters.end(), "--hpx:threads") ==
          parameters.end()) {
        parameters.push_back(const_cast<char *>("--hpx:threads"));
        hpxThreadNum = std::to_string(nHPXThreads);
        parameters.push_back(const_cast<char *>(hpxThreadNum.c_str()));
      }
      parameters.push_back(
          const_cast<char *>("--hpx:ini=hpx.stacks.small_size=0x8000000"));
      parameters.push_back(
          const_cast<char *>("--hpx:ini=hpx.stacks.medium_size=0x10000000"));
      parameters.push_back(
          const_cast<char *>("--hpx:ini=hpx.stacks.large_size=0x20000000"));
      parameters.push_back(
          const_cast<char *>("--hpx:ini=hpx.stacks.huge_size=0x40000000"));
      hpx::start(nullptr, parameters.size(), parameters.data());
    }
  } else {
    hpx::start(nullptr, argc, argv);
  }

  // Instantiate and initialise on each node
  mlir::concretelang::dfr::is_root_node_p =
      (hpx::find_here() == hpx::find_root_locality());
  mlir::concretelang::dfr::num_nodes = hpx::get_num_localities().get();

  new mlir::concretelang::dfr::WorkFunctionRegistry();
  mlir::concretelang::dfr::_dfr_jit_phase_barrier = new hpx::lcos::barrier(
      "phase_barrier", mlir::concretelang::dfr::num_nodes,
      hpx::get_locality_id());
  mlir::concretelang::dfr::_dfr_startup_barrier = new hpx::lcos::barrier(
      "startup_barrier", mlir::concretelang::dfr::num_nodes,
      hpx::get_locality_id());

  if (mlir::concretelang::dfr::_dfr_is_root_node()) {
    // Create compute server components on each node - from the root
    // node only - and the corresponding compute client on the root
    // node.
    mlir::concretelang::dfr::gcc =
        hpx::new_<mlir::concretelang::dfr::GenericComputeClient[]>(
            hpx::default_layout(hpx::find_all_localities()),
            mlir::concretelang::dfr::num_nodes)
            .get();
  }
  END_TIME(&mlir::concretelang::dfr::init_timer, "Initialization");
}

/*  Start/stop functions to be called from within user code (or during
    JIT invocation).  These serve to pause/resume the runtime
    scheduler and to clean up used resources.  */
void _dfr_start(int64_t use_dfr_p, void *ctx) {
  BEGIN_TIME(&mlir::concretelang::dfr::whole_timer);
  if (use_dfr_p) {
    // The first invocation will initialise the runtime. As each call to
    // _dfr_start is matched with _dfr_stop, if this is not hte first,
    // we need to resume the HPX runtime.
    assert(mlir::concretelang::dfr::init_guard !=
               mlir::concretelang::dfr::terminated &&
           "DFR runtime: attempting to start runtime after it has been "
           "terminated");
    uint64_t expected = mlir::concretelang::dfr::uninitialised;
    if (mlir::concretelang::dfr::init_guard.compare_exchange_strong(
            expected, mlir::concretelang::dfr::active))
      _dfr_start_impl(0, nullptr);

    assert(mlir::concretelang::dfr::init_guard ==
               mlir::concretelang::dfr::active &&
           "DFR runtime failed to initialise");

    // If this is not the root node in a non-JIT execution, then this
    // node should only run the scheduler for any incoming work until
    // termination is flagged. If this is JIT, we need to run the
    // cancelled function which registers the work functions.
    if (!mlir::concretelang::dfr::_dfr_is_root_node() &&
        !mlir::concretelang::dfr::_dfr_is_jit())
      _dfr_stop_impl();
  }

  // If DFR is used and a runtime context is needed, and execution is
  // distributed, then broadcast from root to all compute nodes.
  if (use_dfr_p && (mlir::concretelang::dfr::num_nodes > 1) &&
      (ctx || !mlir::concretelang::dfr::_dfr_is_root_node())) {
    BEGIN_TIME(&mlir::concretelang::dfr::broadcast_timer);
    new mlir::concretelang::dfr::RuntimeContextManager();
    mlir::concretelang::dfr::_dfr_node_level_runtime_context_manager
        ->setContext(ctx);

    // If this is not JIT, then the remote nodes never reach _dfr_stop,
    // so root should not instantiate this barrier.
    if (mlir::concretelang::dfr::_dfr_is_root_node() &&
        mlir::concretelang::dfr::_dfr_is_jit())
      mlir::concretelang::dfr::_dfr_startup_barrier->wait();
    END_TIME(&mlir::concretelang::dfr::broadcast_timer, "Key broadcasting");
  }
  BEGIN_TIME(&mlir::concretelang::dfr::compute_timer);
}

// This function cannot be used to terminate the runtime as it is
// non-decidable if another computation phase will follow. Instead the
// _dfr_terminate function provides this facility and is normally
// called on exit from "main" when not using the main wrapper library.
void _dfr_stop(int64_t use_dfr_p) {
  if (use_dfr_p) {
    if (mlir::concretelang::dfr::num_nodes > 1) {
      // Non-root nodes synchronize here with the root to mark the point
      // where the root is free to send work out (only needed in JIT).
      if (!mlir::concretelang::dfr::_dfr_is_root_node())
        mlir::concretelang::dfr::_dfr_startup_barrier->wait();

      // The barrier is only needed to synchronize the different
      // computation phases when the compute nodes need to generate and
      // register new work functions in each phase.

      // TODO: this barrier may be removed based on how work function
      // registration is handled - but it is unlikely to result in much
      // gain as the root node would be waiting for the end of computation
      // on all remote nodes before reaching here anyway (dataflow
      // dependences).
      if (mlir::concretelang::dfr::_dfr_is_jit()) {
        mlir::concretelang::dfr::_dfr_jit_phase_barrier->wait();
      }

      mlir::concretelang::dfr::_dfr_node_level_runtime_context_manager
          ->clearContext();
    }
  }
  END_TIME(&mlir::concretelang::dfr::compute_timer, "Compute");
  END_TIME(&mlir::concretelang::dfr::whole_timer, "Total execution");
}

void _dfr_try_initialize() {
  // Initialize and immediately suspend the HPX runtime if not yet done.
  uint64_t expected = mlir::concretelang::dfr::uninitialised;
  if (mlir::concretelang::dfr::init_guard.compare_exchange_strong(
          expected, mlir::concretelang::dfr::active)) {
    _dfr_start_impl(0, nullptr);
  }

  assert(mlir::concretelang::dfr::init_guard ==
             mlir::concretelang::dfr::active &&
         "DFR runtime failed to initialise");
}

void _dfr_terminate() {
  uint64_t expected = mlir::concretelang::dfr::active;
  if (mlir::concretelang::dfr::init_guard.compare_exchange_strong(
          expected, mlir::concretelang::dfr::terminated))
    _dfr_stop_impl();

  assert((mlir::concretelang::dfr::init_guard ==
              mlir::concretelang::dfr::terminated ||
          mlir::concretelang::dfr::init_guard ==
              mlir::concretelang::dfr::uninitialised) &&
         "DFR runtime failed to terminate");
}

/*******************/
/*  Main wrapper.  */
/*******************/
extern "C" {
extern int main(int argc, char *argv[]) __attribute__((weak));
extern int __real_main(int argc, char *argv[]) __attribute__((weak));
int __wrap_main(int argc, char *argv[]) {
  int r;

  _dfr_try_initialize();
  // Run the actual main function. Within there should be a call to
  // _dfr_start to resume execution of the HPX scheduler if needed.
  r = __real_main(argc, argv);
  _dfr_terminate();

  return r;
}
}

/**********************/
/*  Debug interface.  */
/**********************/
size_t _dfr_debug_get_node_id() { return hpx::get_locality_id(); }

size_t _dfr_debug_get_worker_id() { return hpx::get_worker_thread_num(); }

void _dfr_debug_print_task(const char *name, size_t inputs, size_t outputs) {
  // clang-format off
  hpx::cout << "Task \"" << name << "\t\""
	    << " [" << inputs << " inputs, " << outputs << " outputs]"
	    << "  Executing on Node/Worker: " << _dfr_debug_get_node_id()
	    << " / " << _dfr_debug_get_worker_id() << "\n" << std::flush;
  // clang-format on
}

/// Generic utility function for printing debug info
void _dfr_print_debug(size_t val) {
  hpx::cout << "_dfr_print_debug : " << val << "\n" << std::flush;
}

#else // CONCRETELANG_DATAFLOW_EXECUTION_ENABLED

#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/time_util.h"

namespace mlir {
namespace concretelang {
namespace dfr {
namespace {
static bool is_jit_p = false;
static bool use_omp_p = false;
static size_t num_nodes = 1;
static struct timespec compute_timer;
} // namespace

void _dfr_set_required(bool is_required) {}
void _dfr_set_jit(bool p) { is_jit_p = p; }
void _dfr_set_use_omp(bool use_omp) { use_omp_p = use_omp; }
bool _dfr_is_jit() { return is_jit_p; }
bool _dfr_is_root_node() { return true; }
bool _dfr_use_omp() { return use_omp_p; }
bool _dfr_is_distributed() { return num_nodes > 1; }

} // namespace dfr
} // namespace concretelang
} // namespace mlir

void _dfr_start(int64_t use_dfr_p, void *ctx) {
  BEGIN_TIME(&mlir::concretelang::dfr::compute_timer);
}
void _dfr_stop(int64_t use_dfr_p) {
  END_TIME(&mlir::concretelang::dfr::compute_timer, "Compute");
}

void _dfr_terminate() {}
#endif
