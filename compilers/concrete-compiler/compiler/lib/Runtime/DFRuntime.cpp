// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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
static hpx::distributed::barrier *_dfr_jit_phase_barrier;
static hpx::distributed::barrier *_dfr_startup_barrier;
static size_t num_nodes = 0;
#if CONCRETELANG_TIMING_ENABLED
static struct timespec init_timer, broadcast_timer, compute_timer, whole_timer;
#endif
} // namespace

void *dl_handle = nullptr;
WorkFunctionRegistry *_dfr_node_level_work_function_registry;

} // namespace dfr
} // namespace concretelang
} // namespace mlir

#include "concretelang/Runtime/dfr_tasks.hpp"
using namespace hpx;
using namespace mlir::concretelang::dfr;

// Ready futures are only used as inputs to tasks (never passed to
// await_future), so we only need to track the references in task
// creation.
void *_dfr_make_ready_future(void *in, size_t memref_clone_p) {
  hpx::future<void *> future = hpx::make_ready_future<void *>(in);
  return (void *)new dfr_refcounted_future_t(
      new hpx::shared_future<void *>(std::move(future)), 1, memref_clone_p);
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
  dfr_create_async_task_impl(wfn, ctx, refcounted_futures, param_sizes,
                             param_types, outputs, output_sizes, output_types);
}

/// Runtime generic async_task with vector parameters.  Each first
/// NUM_OUTPUTS quadruplets of arguments in the variadic list
/// corresponds to a size_t for the number of elements in the
/// following array, a void * pointer on an array of
/// hpx::future<void*> and two size_t parameters for the size and type
/// of each output.  After that come NUM_PARAMS quadruplets of
/// arguments in the variadic list that correspond to a size_t for the
/// number of elements in the following array, a void* pointer on an
/// array of hpx::future<void*> and the same two size_t parameters
/// (size and type).
void _dfr_create_async_task_vec(wfnptr wfn, void *ctx, size_t num_params,
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
    size_t count = va_arg(args, uint64_t);
    void **futures = va_arg(args, void **);
    size_t sizes = va_arg(args, uint64_t);
    size_t types = va_arg(args, uint64_t);
    for (size_t j = 0; j < count; ++j) {
      outputs.push_back(futures[j]);
      output_sizes.push_back(sizes);
      output_types.push_back(types);
    }
  }
  for (size_t i = 0; i < num_params; ++i) {
    size_t count = va_arg(args, uint64_t);
    void **futures = va_arg(args, void **);
    size_t sizes = va_arg(args, uint64_t);
    size_t types = va_arg(args, uint64_t);
    for (size_t j = 0; j < count; ++j) {
      refcounted_futures.push_back(futures[j]);
      param_sizes.push_back(sizes);
      param_types.push_back(types);
    }
  }
  va_end(args);

  dfr_create_async_task_impl(wfn, ctx, refcounted_futures, param_sizes,
                             param_types, outputs, output_sizes, output_types);
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
  dfr_required_p = is_required;
  if (dfr_required_p) {
    _dfr_try_initialize();
  }
}
void _dfr_set_jit(bool is_jit) { is_jit_p = is_jit; }
void _dfr_set_use_omp(bool use_omp) { use_omp_p = use_omp; }
bool _dfr_is_jit() { return is_jit_p; }
bool _dfr_is_root_node() { return is_root_node_p; }
bool _dfr_use_omp() { return use_omp_p; }
bool _dfr_is_distributed() { return num_nodes > 1; }
void _dfr_register_lib(void *dlh) { dl_handle = dlh; }
} // namespace dfr
} // namespace concretelang
} // namespace mlir

void _dfr_register_work_function(wfnptr wfn) {
  _dfr_node_level_work_function_registry->getWorkFunctionName((void *)wfn);
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
  if (_dfr_is_root_node())
    hpx::post([]() { hpx::finalize(); });
  hpx::stop();
  exit(EXIT_SUCCESS);
}

static inline void _dfr_start_impl(int argc, char *argv[]) {
  BEGIN_TIME(&init_timer);
  if (dl_handle == nullptr)
    dl_handle = dlopen(nullptr, RTLD_NOW);

  // If OpenMP is to be used, we need to force its initialization
  // before thread binding occurs. Otherwise OMP threads will be bound
  // to the core of the thread initializing the OMP runtime.
  if (_dfr_use_omp()) {
#pragma omp parallel shared(use_omp_p)
    {
#pragma omp critical
      use_omp_p = true;
    }
  }

  if (argc == 0) {
    int nCores, nOMPThreads, nHPXThreads;
    std::string hpxThreadNum;

    std::vector<char *> parameters;
    parameters.push_back(const_cast<char *>("__dummy_dfr_HPX_program_name__"));

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
    if (_dfr_use_omp() && env != nullptr)
      nOMPThreads = strtoul(env, NULL, 10);
    else if (_dfr_use_omp())
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
  is_root_node_p = (hpx::find_here() == hpx::find_root_locality());
  num_nodes = hpx::get_num_localities().get();

  new WorkFunctionRegistry();

  char *env = getenv("DFR_LAZY_KEY_TRANSFER");
  bool lazy = false;
  if (env != nullptr)
    if (!strncmp(env, "True", 4) || !strncmp(env, "true", 4) ||
        !strncmp(env, "On", 2) || !strncmp(env, "on", 2) ||
        !strncmp(env, "1", 1))
      lazy = true;
  new RuntimeContextManager(lazy);

  _dfr_jit_phase_barrier = new hpx::distributed::barrier(
      "phase_barrier", num_nodes, hpx::get_locality_id());
  _dfr_startup_barrier = new hpx::distributed::barrier(
      "startup_barrier", num_nodes, hpx::get_locality_id());

  if (_dfr_is_root_node()) {
    // Create compute server components on each node - from the root
    // node only - and the corresponding compute client on the root
    // node.
    gcc = hpx::new_<GenericComputeClient[]>(
              hpx::default_layout(hpx::find_all_localities()), num_nodes)
              .get();
  }
  END_TIME(&init_timer, "Initialization");
}

/*  Start/stop functions to be called from within user code (or during
    JIT invocation).  These serve to pause/resume the runtime
    scheduler and to clean up used resources.  */
void _dfr_start(int64_t use_dfr_p, void *ctx) {
  BEGIN_TIME(&whole_timer);
  if (use_dfr_p) {
    // The first invocation will initialise the runtime. As each call to
    // _dfr_start is matched with _dfr_stop, if this is not the first,
    // we need to resume the HPX runtime.
    assert(init_guard != terminated &&
           "DFR runtime: attempting to start runtime after it has been "
           "terminated");
    uint64_t expected = uninitialised;
    if (init_guard.compare_exchange_strong(expected, active))
      _dfr_start_impl(0, nullptr);

    assert(init_guard == active && "DFR runtime failed to initialise");

    // If execution is distributed, then broadcast (possibly an empty)
    // context from root to all compute nodes.
    if (num_nodes > 1) {
      BEGIN_TIME(&broadcast_timer);
      _dfr_node_level_runtime_context_manager->setContext(ctx);
      _dfr_startup_barrier->wait();
      if (ctx) {
        END_TIME(&broadcast_timer, "Key broadcasting");
      }
    }
  }
  BEGIN_TIME(&compute_timer);
}

// This function cannot be used to terminate the runtime as it is
// non-decidable if another computation phase will follow. Instead the
// _dfr_terminate function provides this facility and is normally
// called on exit from "main" when not using the main wrapper library.
void _dfr_stop(int64_t use_dfr_p) {
  if (use_dfr_p) {
    if (num_nodes > 1) {
      // The barrier is only needed to synchronize the different
      // computation phases when the compute nodes need to generate and
      // register new work functions in each phase.
      _dfr_jit_phase_barrier->wait();
      _dfr_node_level_runtime_context_manager->clearContext();
      _dfr_node_level_work_function_registry->clearRegistry();
      _dfr_jit_phase_barrier->wait();
    }
  }
  END_TIME(&compute_timer, "Compute");
  END_TIME(&whole_timer, "Total execution");
}

namespace mlir {
namespace concretelang {
namespace dfr {
void _dfr_run_remote_scheduler() {
  _dfr_start(1, nullptr);
  _dfr_stop(1);
}
} // namespace dfr
} // namespace concretelang
} // namespace mlir
void _dfr_try_initialize() {
  // Initialize and immediately suspend the HPX runtime if not yet done.
  uint64_t expected = uninitialised;
  if (init_guard.compare_exchange_strong(expected, active)) {
    _dfr_start_impl(0, nullptr);
  }

  assert(init_guard == active && "DFR runtime failed to initialise");
}

void _dfr_terminate() {
  uint64_t expected = active;
  if (init_guard.compare_exchange_strong(expected, terminated))
    _dfr_stop_impl();

  assert((init_guard == terminated || init_guard == uninitialised) &&
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
#if CONCRETELANG_TIMING_ENABLED
static struct timespec compute_timer;
#endif
} // namespace

void _dfr_set_required(bool is_required) {}
void _dfr_set_jit(bool p) { is_jit_p = p; }
void _dfr_set_use_omp(bool use_omp) { use_omp_p = use_omp; }
bool _dfr_is_jit() { return is_jit_p; }
bool _dfr_is_root_node() { return true; }
bool _dfr_use_omp() { return use_omp_p; }
bool _dfr_is_distributed() { return num_nodes > 1; }
void _dfr_run_remote_scheduler() {}
void _dfr_register_lib(void *dlh) {}

} // namespace dfr
} // namespace concretelang
} // namespace mlir

using namespace mlir::concretelang::dfr;

void _dfr_start(int64_t use_dfr_p, void *ctx) { BEGIN_TIME(&compute_timer); }
void _dfr_stop(int64_t use_dfr_p) { END_TIME(&compute_timer, "Compute"); }

void _dfr_terminate() {}
#endif
