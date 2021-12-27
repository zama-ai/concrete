// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

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

#include "zamalang/Runtime/DFRuntime.hpp"
#include "zamalang/Runtime/distributed_generic_task_server.hpp"
#include "zamalang/Runtime/runtime_api.h"

std::vector<GenericComputeClient> gcc;
void *dl_handle;
PbsKeyManager *node_level_key_manager;
WorkFunctionRegistry *node_level_work_function_registry;
std::list<void *> new_allocated;
std::list<void *> fut_allocated;
std::list<void *> m_allocated;

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

/* Distributed key management.  */
void _dfr_register_key(void *key, size_t key_id, size_t size) {
  node_level_key_manager->register_key(key, key_id, size);
}

void _dfr_broadcast_keys() { node_level_key_manager->broadcast_keys(); }

void *_dfr_get_key(size_t key_id) {
  return *node_level_key_manager->get_key(key_id).key.get();
}

/* Runtime initialization and finalization.  */
static inline void _dfr_stop_impl() {
  hpx::apply([]() { hpx::finalize(); });
  hpx::stop();
}

static inline void _dfr_start_impl(int argc, char *argv[]) {
  dl_handle = dlopen(nullptr, RTLD_NOW);
  if (argc == 0) {
    char *_argv[1] = {const_cast<char *>("__dummy_dfr_HPX_program_name__")};
    int _argc = 1;
    hpx::start(nullptr, _argc, _argv);
  } else {
    hpx::start(nullptr, argc, argv);
  }

  new PbsKeyManager();
  new WorkFunctionRegistry();

  if (!_dfr_is_root_node()) {
    _dfr_stop_impl();
    exit(EXIT_SUCCESS);
  }

  // Create compute server components on each node and the
  // corresponding compute client.
  auto num_nodes = hpx::get_num_localities().get();
  gcc = hpx::new_<GenericComputeClient[]>(
            hpx::default_layout(hpx::find_all_localities()), num_nodes)
            .get();
}

// TODO: we need a better way to wrap main. For now loader --wrap and
// main's constructor/destructor are not functional, but that should
// replace the current, inefficient calls to _dfr_start/stop generated
// in each compiled function.
void _dfr_start() {
  static int first_p = 0;
  if (!first_p) {
    _dfr_start_impl(0, nullptr);
    first_p = 1;
  } else {
    hpx::resume();
  }
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

/*  Debug interface.  */
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
