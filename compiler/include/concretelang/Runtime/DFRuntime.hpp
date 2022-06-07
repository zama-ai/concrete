// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_DFRUNTIME_HPP
#define CONCRETELANG_DFR_DFRUNTIME_HPP

#include <dlfcn.h>
#include <memory>
#include <utility>

#include "concretelang/Runtime/runtime_api.h"

/*  Debug interface.  */
#include "concretelang/Runtime/dfr_debug_interface.h"

extern void *dl_handle;
struct WorkFunctionRegistry;
extern WorkFunctionRegistry *node_level_work_function_registry;

// Recover the name of the work function
static inline const char *_dfr_get_function_name_from_address(void *fn) {
  Dl_info info;

  if (!dladdr(fn, &info) || info.dli_sname == nullptr)
    HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_get_function_name_from_address",
                        "Error recovering work function name from address.");
  return info.dli_sname;
}

static inline wfnptr _dfr_get_function_pointer_from_name(const char *fn_name) {
  auto ptr = dlsym(dl_handle, fn_name);

  if (ptr == nullptr)
    HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_get_function_pointer_from_name",
                        "Error recovering work function pointer from name.");
  return (wfnptr)ptr;
}

// Determine where new task should run.  For now just round-robin
// distribution - TODO: optimise.
static inline size_t _dfr_find_next_execution_locality() {
  static size_t num_nodes = hpx::get_num_localities().get();
  static std::atomic<std::size_t> next_locality{0};

  size_t next_loc = ++next_locality;

  return next_loc % num_nodes;
}

static inline bool _dfr_is_root_node() {
  return hpx::find_here() == hpx::find_root_locality();
}

struct WorkFunctionRegistry {
  WorkFunctionRegistry() { node_level_work_function_registry = this; }

  wfnptr getWorkFunctionPointer(const std::string &name) {
    std::lock_guard<std::mutex> guard(registry_guard);

    auto fnptrit = name_to_ptr_registry.find(name);
    if (fnptrit != name_to_ptr_registry.end())
      return (wfnptr)fnptrit->second;

    auto ptr = dlsym(dl_handle, name.c_str());
    if (ptr == nullptr)
      HPX_THROW_EXCEPTION(hpx::no_success,
                          "WorkFunctionRegistry::getWorkFunctionPointer",
                          "Error recovering work function pointer from name.");
    ptr_to_name_registry.insert(
        std::pair<const void *, std::string>(ptr, name));
    name_to_ptr_registry.insert(
        std::pair<std::string, const void *>(name, ptr));
    return (wfnptr)ptr;
  }

  std::string getWorkFunctionName(const void *fn) {
    std::lock_guard<std::mutex> guard(registry_guard);

    auto fnnameit = ptr_to_name_registry.find(fn);
    if (fnnameit != ptr_to_name_registry.end())
      return fnnameit->second;

    Dl_info info;
    std::string ret;
    // Assume that if we can't find the name, there is no dynamic
    // library to find it in. TODO: fix this to distinguish JIT/binary
    // and in case of distributed exec.
    if (!dladdr(fn, &info) || info.dli_sname == nullptr) {
      static std::atomic<unsigned int> fnid{0};
      ret = "_dfr_jit_wfnname_" + std::to_string(fnid++);
    } else {
      ret = info.dli_sname;
    }
    ptr_to_name_registry.insert(std::pair<const void *, std::string>(fn, ret));
    name_to_ptr_registry.insert(std::pair<std::string, const void *>(ret, fn));
    return ret;
  }

private:
  std::mutex registry_guard;
  std::map<const void *, std::string> ptr_to_name_registry;
  std::map<std::string, const void *> name_to_ptr_registry;
};

#endif
