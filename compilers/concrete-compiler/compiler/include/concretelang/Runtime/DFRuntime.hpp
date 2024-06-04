// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_DFRUNTIME_HPP
#define CONCRETELANG_DFR_DFRUNTIME_HPP

#include <cassert>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <utility>

#include "concretelang/Runtime/runtime_api.h"

namespace mlir {
namespace concretelang {
namespace dfr {

void _dfr_set_required(bool);
void _dfr_set_jit(bool);
void _dfr_set_use_omp(bool);
bool _dfr_is_jit();
bool _dfr_is_root_node();
bool _dfr_use_omp();
bool _dfr_is_distributed();
void _dfr_run_remote_scheduler();
void _dfr_register_lib(void *dlh);

typedef enum _dfr_task_arg_type {
  _DFR_TASK_ARG_BASE = 0,
  _DFR_TASK_ARG_MEMREF = 1,
  _DFR_TASK_ARG_CONTEXT = 2
} _dfr_task_arg_type;

static inline _dfr_task_arg_type _dfr_get_arg_type(uint64_t val) {
  return (_dfr_task_arg_type)(val & 0xFF);
}
static inline uint64_t _dfr_get_memref_element_size(uint64_t val) {
  return val >> 8;
}
static inline uint64_t _dfr_set_arg_type(uint64_t val,
                                         _dfr_task_arg_type type) {
  return (val & ~(0xFF)) | type;
}
static inline uint64_t _dfr_set_memref_element_size(uint64_t val, size_t size) {
  assert(size < (((uint64_t)1) << 48));
  return (val & 0xFF) | (((uint64_t)size) << 8);
}

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif
