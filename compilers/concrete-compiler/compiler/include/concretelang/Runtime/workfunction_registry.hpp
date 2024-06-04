// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_WORKFUNCTION_REGISTRY_HPP
#define CONCRETELANG_DFR_WORKFUNCTION_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <utility>

#include <hpx/include/runtime.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>

#include "concretelang/Runtime/DFRuntime.hpp"

namespace mlir {
namespace concretelang {
namespace dfr {

struct WorkFunctionRegistry;
extern WorkFunctionRegistry *_dfr_node_level_work_function_registry;
extern void *dl_handle;

struct WorkFunctionRegistry {
  WorkFunctionRegistry() { _dfr_node_level_work_function_registry = this; }

  wfnptr getWorkFunctionPointer(const std::string &name) {
    std::lock_guard<std::mutex> guard(registry_guard);

    auto fnptrit = name_to_ptr_registry.find(name);
    if (fnptrit != name_to_ptr_registry.end())
      return (wfnptr)fnptrit->second;

    auto ptr = dlsym(dl_handle, name.c_str());
    if (ptr == nullptr) {
      HPX_THROW_EXCEPTION(hpx::error::no_success,
                          "WorkFunctionRegistry::getWorkFunctionPointer",
                          "Error recovering work function pointer from name.");
    }
    registerWorkFunction(ptr, name);
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
      ret = registerAnonymousWorkFunction(fn);
    } else {
      ret = info.dli_sname;
      registerWorkFunction(fn, ret);
    }
    return ret;
  }

  void clearRegistry() {
    std::lock_guard<std::mutex> guard(registry_guard);

    ptr_to_name_registry.clear();
    name_to_ptr_registry.clear();
    fnid = 0;
  }

private:
  void registerWorkFunction(const void *fn, std::string name) {

    auto fnnameit = ptr_to_name_registry.find(fn);
    if (fnnameit == ptr_to_name_registry.end())
      ptr_to_name_registry.insert(
          std::pair<const void *, std::string>(fn, name));

    auto fnptrit = name_to_ptr_registry.find(name);
    if (fnptrit == name_to_ptr_registry.end())
      name_to_ptr_registry.insert(
          std::pair<std::string, const void *>(name, fn));
  }

  std::string registerAnonymousWorkFunction(const void *fn) {
    std::string name = "_dfr_jit_wfnname_" + std::to_string(fnid++);
    registerWorkFunction(fn, name);
    return name;
  }

private:
  std::mutex registry_guard;
  std::atomic<unsigned int> fnid{0};
  std::map<const void *, std::string> ptr_to_name_registry;
  std::map<std::string, const void *> name_to_ptr_registry;
};

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif
