// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

/**
   This file implements the dataflow runtime. It encapsulates all of
   the underlying communication, parallelism, etc. and only exposes a
   simplified interface for code generation in runtime_api.h

   This hides the details of implementation, including of the HPX
   framework currently used, from the code generation side.
 */

#include <hpx/future.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hwloc.h>

#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/distributed_generic_task_server.hpp"
#include "concretelang/Runtime/runtime_api.h"

std::vector<GenericComputeClient> gcc;
void *dl_handle;
PbsKeyManager *node_level_key_manager;
WorkFunctionRegistry *node_level_work_function_registry;
std::list<void *> new_allocated;
std::list<void *> fut_allocated;
std::list<void *> m_allocated;
std::atomic<uint64_t> init_guard = {0};

using namespace hpx;

void *_dfr_make_ready_future(void *in) {
  void *future = static_cast<void *>(
      new hpx::shared_future<void *>(hpx::make_ready_future(in)));
  m_allocated.push_back(in);
  fut_allocated.push_back(future);
  return future;
}

void *_dfr_await_future(void *in) {
  return static_cast<hpx::shared_future<void *> *>(in)->get();
}

void _dfr_deallocate_future_data(void *in) {
  delete[] static_cast<char *>(
      static_cast<hpx::shared_future<void *> *>(in)->get());
}

void _dfr_deallocate_future(void *in) {
  delete (static_cast<hpx::shared_future<void *> *>(in));
}

// Runtime generic async_task.  Each first NUM_PARAMS pairs of
// arguments in the variadic list corresponds to a void* pointer on a
// hpx::future<void*> and the size of data within the future.  After
// that come NUM_OUTPUTS pairs of hpx::future<void*>* and size_t for
// the returns.
void _dfr_create_async_task(wfnptr wfn, size_t num_params, size_t num_outputs,
                            ...) {
  std::vector<void *> params;
  std::vector<void *> outputs;
  std::vector<size_t> param_sizes;
  std::vector<size_t> output_sizes;

  va_list args;
  va_start(args, num_outputs);
  for (size_t i = 0; i < num_params; ++i) {
    params.push_back(va_arg(args, void *));
    param_sizes.push_back(va_arg(args, size_t));
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    outputs.push_back(va_arg(args, void *));
    output_sizes.push_back(va_arg(args, size_t));
  }
  va_end(args);

  // We pass functions by name - which is not strictly necessary in
  // shared memory as pointers suffice, but is needed in the
  // distributed case where the functions need to be located/loaded on
  // the node.
  auto wfnname =
      node_level_work_function_registry->getWorkFunctionName((void *)wfn);
  hpx::future<hpx::future<OpaqueOutputData>> oodf;

  // In order to allow complete dataflow semantics for
  // communication/synchronization, we split tasks in two parts: an
  // execution body that is scheduled once all input dependences are
  // satisfied, which generates a future on a tuple of outputs, which
  // is then further split into a tuple of futures and provide
  // individual synchronization for each return independently.
  switch (num_params) {
  case 0:
    oodf = std::move(
        hpx::dataflow([wfnname, param_sizes,
                       output_sizes]() -> hpx::future<OpaqueOutputData> {
          std::vector<void *> params = {};
          OpaqueInputData oid(wfnname, params, param_sizes, output_sizes);
          return gcc[_dfr_find_next_execution_locality()].execute_task(oid);
        }));
    break;

  case 1:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, output_sizes](hpx::shared_future<void *> param0)
            -> hpx::future<OpaqueOutputData> {
          std::vector<void *> params = {param0.get()};
          OpaqueInputData oid(wfnname, params, param_sizes, output_sizes);
          return gcc[_dfr_find_next_execution_locality()].execute_task(oid);
        },
        *(hpx::shared_future<void *> *)params[0]));
    break;

  case 2:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, output_sizes](hpx::shared_future<void *> param0,
                                             hpx::shared_future<void *> param1)
            -> hpx::future<OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get()};
          OpaqueInputData oid(wfnname, params, param_sizes, output_sizes);
          return gcc[_dfr_find_next_execution_locality()].execute_task(oid);
        },
        *(hpx::shared_future<void *> *)params[0],
        *(hpx::shared_future<void *> *)params[1]));
    break;

  case 3:
    oodf = std::move(hpx::dataflow(
        [wfnname, param_sizes, output_sizes](hpx::shared_future<void *> param0,
                                             hpx::shared_future<void *> param1,
                                             hpx::shared_future<void *> param2)
            -> hpx::future<OpaqueOutputData> {
          std::vector<void *> params = {param0.get(), param1.get(),
                                        param2.get()};
          OpaqueInputData oid(wfnname, params, param_sizes, output_sizes);
          return gcc[_dfr_find_next_execution_locality()].execute_task(oid);
        },
        *(hpx::shared_future<void *> *)params[0],
        *(hpx::shared_future<void *> *)params[1],
        *(hpx::shared_future<void *> *)params[2]));
    break;

  default:
    HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_create_async_task",
                        "Error: number of task parameters not supported.");
  }

  switch (num_outputs) {
  case 1:
    *((void **)outputs[0]) = new hpx::shared_future<void *>(hpx::dataflow(
        [](hpx::future<OpaqueOutputData> oodf_in) -> void * {
          return oodf_in.get().outputs[0];
        },
        oodf));
    fut_allocated.push_back(*((void **)outputs[0]));
    break;

  case 2: {
    hpx::future<hpx::tuple<void *, void *>> &&ft = hpx::dataflow(
        [](hpx::future<OpaqueOutputData> oodf_in)
            -> hpx::tuple<void *, void *> {
          std::vector<void *> outputs = std::move(oodf_in.get().outputs);
          return hpx::make_tuple<>(outputs[0], outputs[1]);
        },
        oodf);
    hpx::tuple<hpx::future<void *>, hpx::future<void *>> &&tf =
        hpx::split_future(std::move(ft));
    *((void **)outputs[0]) =
        (void *)new hpx::shared_future<void *>(std::move(hpx::get<0>(tf)));
    *((void **)outputs[1]) =
        (void *)new hpx::shared_future<void *>(std::move(hpx::get<1>(tf)));
    fut_allocated.push_back(*((void **)outputs[0]));
    fut_allocated.push_back(*((void **)outputs[1]));
    break;
  }

  case 3: {
    hpx::future<hpx::tuple<void *, void *, void *>> &&ft = hpx::dataflow(
        [](hpx::future<OpaqueOutputData> oodf_in)
            -> hpx::tuple<void *, void *, void *> {
          std::vector<void *> outputs = std::move(oodf_in.get().outputs);
          return hpx::make_tuple<>(outputs[0], outputs[1], outputs[2]);
        },
        oodf);
    hpx::tuple<hpx::future<void *>, hpx::future<void *>, hpx::future<void *>>
        &&tf = hpx::split_future(std::move(ft));
    *((void **)outputs[0]) =
        (void *)new hpx::shared_future<void *>(std::move(hpx::get<0>(tf)));
    *((void **)outputs[1]) =
        (void *)new hpx::shared_future<void *>(std::move(hpx::get<1>(tf)));
    *((void **)outputs[2]) =
        (void *)new hpx::shared_future<void *>(std::move(hpx::get<2>(tf)));
    fut_allocated.push_back(*((void **)outputs[0]));
    fut_allocated.push_back(*((void **)outputs[1]));
    fut_allocated.push_back(*((void **)outputs[2]));
    break;
  }
  default:
    HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_create_async_task",
                        "Error: number of task outputs not supported.");
  }
}

/********************************/
/* Distributed key management.  */
/********************************/
void _dfr_register_key(void *key, size_t key_id, size_t size) {
  node_level_key_manager->register_key(key, key_id, size);
}

void _dfr_broadcast_keys() { node_level_key_manager->broadcast_keys(); }

void *_dfr_get_key(size_t key_id) {
  return *node_level_key_manager->get_key(key_id).key.get();
}

/************************************/
/*  Initialization & Finalization.  */
/************************************/
/* Runtime initialization and finalization.  */
static inline void _dfr_stop_impl() {
  if (_dfr_is_root_node())
    hpx::apply([]() { hpx::finalize(); });
  hpx::stop();
}

static inline void _dfr_start_impl(int argc, char *argv[]) {
  dl_handle = dlopen(nullptr, RTLD_NOW);
  if (argc == 0) {
    unsigned long nCores, nOMPThreads, nHPXThreads;
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_set_all_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_NONE);
    hwloc_topology_set_type_filter(topology, HWLOC_OBJ_CORE,
                                   HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topology);
    nCores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    if (nCores < 1)
      nCores = 1;
    nOMPThreads = 1;
    char *env = getenv("OMP_NUM_THREADS");
    if (env != nullptr) {
      nOMPThreads = strtoul(env, NULL, 10);
      if (nOMPThreads == 0)
        nOMPThreads = 1;
      if (nOMPThreads >= nCores)
        nOMPThreads = nCores;
    }
    std::string numOMPThreads = std::to_string(nOMPThreads);
    setenv("OMP_NUM_THREADS", numOMPThreads.c_str(), 0);
    nHPXThreads = nCores + 1 - nOMPThreads;
    std::string numHPXThreads = std::to_string(nHPXThreads);
    char *_argv[3] = {const_cast<char *>("__dummy_dfr_HPX_program_name__"),
                      const_cast<char *>("--hpx:threads"),
                      const_cast<char *>(numHPXThreads.c_str())};
    int _argc = 3;
    hpx::start(nullptr, _argc, _argv);
  } else {
    hpx::start(nullptr, argc, argv);
  }

  // Instantiate on each node
  new PbsKeyManager();
  new WorkFunctionRegistry();

  if (_dfr_is_root_node()) {
    // Create compute server components on each node - from the root
    // node only - and the corresponding compute client on the root
    // node.
    auto num_nodes = hpx::get_num_localities().get();
    gcc = hpx::new_<GenericComputeClient[]>(
              hpx::default_layout(hpx::find_all_localities()), num_nodes)
              .get();
  } else {
    hpx::stop();
    exit(EXIT_SUCCESS);
  }
}

/*  Start/stop functions to be called from within user code (or during
    JIT invocation).  These serve to pause/resume the runtime
    scheduler and to clean up used resources.  */
void _dfr_start() {
  uint64_t uninitialised = 0;
  if (init_guard.compare_exchange_strong(uninitialised, 1))
    _dfr_start_impl(0, nullptr);
  else
    hpx::resume();
}

void _dfr_stop() {
  hpx::suspend();

  while (!new_allocated.empty()) {
    delete[] static_cast<char *>(new_allocated.front());
    new_allocated.pop_front();
  }
  while (!fut_allocated.empty()) {
    delete static_cast<hpx::shared_future<void *> *>(fut_allocated.front());
    fut_allocated.pop_front();
  }
  while (!m_allocated.empty()) {
    free(m_allocated.front());
    m_allocated.pop_front();
  }
}

void _dfr_terminate() {
  uint64_t initialised = 1;
  if (init_guard.compare_exchange_strong(initialised, 2)) {
    hpx::resume();
    _dfr_stop_impl();
  }
}

/*******************/
/*  Main wrapper.  */
/*******************/
extern "C" {
extern int main(int argc, char *argv[]); // __attribute__((weak));
extern int __real_main(int argc, char *argv[]) __attribute__((weak));
int __wrap_main(int argc, char *argv[]) {
  int r;
  // Initialize and immediately suspend the HPX runtime if not yet done.
  uint64_t uninitialised = 0;
  if (init_guard.compare_exchange_strong(uninitialised, 1)) {
    _dfr_start_impl(0, nullptr);
    hpx::suspend();
  }
  // Run the actual main function. Within there should be a call to
  // _dfr_start to resume execution of the HPX scheduler if needed.
  r = __real_main(argc, argv);
  // By default all _dfr_start should be matched to a _dfr_stop, so we
  // need to resume before being able to finalize.
  uint64_t initialised = 1;
  if (init_guard.compare_exchange_strong(initialised, 2)) {
    hpx::resume();
    _dfr_stop_impl();
  }

  return r;
}
}

/**********************/
/*  Debug interface.  */
/**********************/
size_t _dfr_debug_get_node_id() { return hpx::get_locality_id(); }

size_t _dfr_debug_get_worker_id() { return hpx::get_worker_thread_num(); }

void _dfr_debug_print_task(const char *name, int inputs, int outputs) {
  // clang-format off
  hpx::cout << "Task \"" << name << "\""
	    << " [" << inputs << " inputs, " << outputs << " outputs]"
	    << "  Executing on Node/Worker: " << _dfr_debug_get_node_id()
	    << " / " << _dfr_debug_get_worker_id() << "\n" << std::flush;
  // clang-format on
}

// Generic utility function for printing debug info
void _dfr_print_debug(size_t val) {
  hpx::cout << "_dfr_print_debug : " << val << "\n" << std::flush;
}
