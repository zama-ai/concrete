// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_DISTRIBUTED_GENERIC_TASK_SERVER_HPP
#define CONCRETELANG_DFR_DISTRIBUTED_GENERIC_TASK_SERVER_HPP

#include <cstdarg>
#include <cstdlib>
#include <malloc.h>
#include <string>

#include <hpx/async_colocated/get_colocation_id.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/util.hpp>
#include <hpx/iostream.hpp>
#include <hpx/serialization/detail/serialize_collection.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>

#include <hpx/async_colocated/get_colocation_id.hpp>
#include <hpx/include/client.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/collectives.hpp>

#include <mlir/ExecutionEngine/CRunnerUtils.h>

#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/context.h"
#include "concretelang/Runtime/dfr_debug_interface.h"
#include "concretelang/Runtime/key_manager.hpp"
#include "concretelang/Runtime/runtime_api.h"
#include "concretelang/Runtime/workfunction_registry.hpp"

using namespace hpx::components;
using namespace hpx::collectives;

namespace mlir {
namespace concretelang {
namespace dfr {

static inline size_t _dfr_get_memref_rank(size_t size) {
  return (size - 2 * sizeof(char *) /*allocated_ptr & aligned_ptr*/
          - sizeof(int64_t) /*offset*/) /
         (2 * sizeof(int64_t) /*size&stride/rank*/);
}

static inline void _dfr_checked_aligned_alloc(void **out, size_t align,
                                              size_t size) {
  int res = posix_memalign(out, align, size);
  if (res == ENOMEM)
    HPX_THROW_EXCEPTION(hpx::error::no_success, "DFR: memory allocation failed",
                        "Error: insufficient memory available.");
  if (res == EINVAL)
    HPX_THROW_EXCEPTION(hpx::error::no_success, "DFR: memory allocation failed",
                        "Error: invalid memory alignment.");
}

struct OpaqueInputData {
  OpaqueInputData() = default;

  OpaqueInputData(std::string _wfn_name, std::vector<void *> _params,
                  std::vector<size_t> _param_sizes,
                  std::vector<uint64_t> _param_types,
                  std::vector<size_t> _output_sizes,
                  std::vector<uint64_t> _output_types, void *_context = nullptr)
      : wfn_name(_wfn_name), params(std::move(_params)),
        param_sizes(std::move(_param_sizes)),
        param_types(std::move(_param_types)),
        output_sizes(std::move(_output_sizes)),
        output_types(std::move(_output_types)), context(_context) {
    if (_context)
      params.push_back(_context);
  }

  OpaqueInputData(const OpaqueInputData &oid)
      : wfn_name(std::move(oid.wfn_name)), params(std::move(oid.params)),
        param_sizes(std::move(oid.param_sizes)),
        param_types(std::move(oid.param_types)),
        output_sizes(std::move(oid.output_sizes)),
        output_types(std::move(oid.output_types)), context(oid.context) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    bool has_context;
    ar >> wfn_name >> has_context;
    ar >> param_sizes >> param_types;
    ar >> output_sizes >> output_types;
    for (size_t p = 0; p < param_sizes.size(); ++p) {
      char *param;
      _dfr_checked_aligned_alloc((void **)&param, sizeof(void *),
                                 param_sizes[p]);
      ar >> hpx::serialization::make_array(param, param_sizes[p]);
      params.push_back((void *)param);

      switch (_dfr_get_arg_type(param_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(param_sizes[p]);
        UnrankedMemRefType<char> umref = {(int64_t)rank, params[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(param_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        size_t alloc_size = (size + mref.offset) * elementSize;
        char *data;
        _dfr_checked_aligned_alloc((void **)&data, 512, alloc_size);
        ar >> hpx::serialization::make_array(data + mref.offset * elementSize,
                                             size * elementSize);
        static_cast<StridedMemRefType<char, 1> *>(params[p])->basePtr = nullptr;
        static_cast<StridedMemRefType<char, 1> *>(params[p])->data = data;
      } break;
      default:
        HPX_THROW_EXCEPTION(hpx::error::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
    if (has_context)
      params.push_back(
          (void *)_dfr_node_level_runtime_context_manager->getContext());
  }
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    bool has_context = (bool)(context != nullptr);
    ar << wfn_name << has_context;
    ar << param_sizes << param_types;
    ar << output_sizes << output_types;
    for (size_t p = 0; p < param_sizes.size(); ++p) {
      // Save the first level of the data structure - if the parameter
      // is a tensor/memref, there is a second level.
      ar << hpx::serialization::make_array((char *)params[p], param_sizes[p]);
      switch (_dfr_get_arg_type(param_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(param_sizes[p]);
        UnrankedMemRefType<char> umref = {(int64_t)rank, params[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(param_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        ar << hpx::serialization::make_array(
            mref.data + mref.offset * elementSize, size * elementSize);
      } break;
      default:
        HPX_THROW_EXCEPTION(hpx::error::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

  std::string wfn_name;
  std::vector<void *> params;
  std::vector<size_t> param_sizes;
  std::vector<uint64_t> param_types;
  std::vector<size_t> output_sizes;
  std::vector<uint64_t> output_types;
  void *context;
};

struct OpaqueOutputData {
  OpaqueOutputData() = default;
  OpaqueOutputData(std::vector<void *> outputs,
                   std::vector<size_t> output_sizes,
                   std::vector<uint64_t> output_types)
      : outputs(std::move(outputs)), output_sizes(std::move(output_sizes)),
        output_types(std::move(output_types)) {}
  OpaqueOutputData(const OpaqueOutputData &ood)
      : outputs(std::move(ood.outputs)),
        output_sizes(std::move(ood.output_sizes)),
        output_types(std::move(ood.output_types)) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar >> output_sizes >> output_types;
    for (size_t p = 0; p < output_sizes.size(); ++p) {
      char *output;
      _dfr_checked_aligned_alloc((void **)&output, sizeof(void *),
                                 (output_sizes[p]));

      ar >> hpx::serialization::make_array(output, output_sizes[p]);
      outputs.push_back((void *)output);

      switch (_dfr_get_arg_type(output_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(output_sizes[p]);
        UnrankedMemRefType<char> umref = {(int64_t)rank, outputs[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(output_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        size_t alloc_size = (size + mref.offset) * elementSize;
        char *data;
        _dfr_checked_aligned_alloc((void **)&data, 512, alloc_size);
        ar >> hpx::serialization::make_array(data + mref.offset * elementSize,
                                             size * elementSize);
        static_cast<StridedMemRefType<char, 1> *>(outputs[p])->basePtr =
            nullptr;
        static_cast<StridedMemRefType<char, 1> *>(outputs[p])->data = data;
      } break;
      default:
        HPX_THROW_EXCEPTION(hpx::error::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
  }
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    ar << output_sizes << output_types;
    for (size_t p = 0; p < outputs.size(); ++p) {
      ar << hpx::serialization::make_array((char *)outputs[p], output_sizes[p]);

      switch (_dfr_get_arg_type(output_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(output_sizes[p]);
        UnrankedMemRefType<char> umref = {(int64_t)rank, outputs[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(output_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        ar << hpx::serialization::make_array(
            mref.data + mref.offset * elementSize, size * elementSize);
      } break;
      default:
        HPX_THROW_EXCEPTION(hpx::error::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

  std::vector<void *> outputs;
  std::vector<size_t> output_sizes;
  std::vector<uint64_t> output_types;
};

struct GenericComputeServer : component_base<GenericComputeServer> {
  GenericComputeServer() = default;

  // Component actions exposed
  OpaqueOutputData execute_task(const OpaqueInputData &inputs) {
    auto wfn = _dfr_node_level_work_function_registry->getWorkFunctionPointer(
        inputs.wfn_name);
    std::vector<void *> outputs;

    switch (inputs.output_sizes.size()) {

#include "concretelang/Runtime/generated/dfr_task_work_function_calls.h"

    default:
      HPX_THROW_EXCEPTION(hpx::error::no_success,
                          "GenericComputeServer::execute_task",
                          "Error: number of task outputs not supported.");
    }

    // Deallocate input data buffers from OID deserialization (load)
    if (!_dfr_is_root_node()) {
      for (size_t p = 0; p < inputs.param_sizes.size(); ++p) {
        if (_dfr_get_arg_type(inputs.param_types[p]) == _DFR_TASK_ARG_MEMREF)
          delete (static_cast<StridedMemRefType<char, 1> *>(inputs.params[p])
                      ->data);
        delete ((char *)inputs.params[p]);
      }
    }

    return OpaqueOutputData(std::move(outputs), std::move(inputs.output_sizes),
                            std::move(inputs.output_types));
  }

  HPX_DEFINE_COMPONENT_ACTION(GenericComputeServer, execute_task)
};

} // namespace dfr
} // namespace concretelang
} // namespace mlir

HPX_REGISTER_ACTION_DECLARATION(
    mlir::concretelang::dfr::GenericComputeServer::execute_task_action,
    GenericComputeServer_execute_task_action)

HPX_REGISTER_COMPONENT_MODULE()
HPX_REGISTER_COMPONENT(
    hpx::components::component<mlir::concretelang::dfr::GenericComputeServer>,
    GenericComputeServer)

HPX_REGISTER_ACTION(
    mlir::concretelang::dfr::GenericComputeServer::execute_task_action,
    GenericComputeServer_execute_task_action)

namespace mlir {
namespace concretelang {
namespace dfr {

struct GenericComputeClient
    : client_base<GenericComputeClient, GenericComputeServer> {
  typedef client_base<GenericComputeClient, GenericComputeServer> base_type;

  GenericComputeClient() = default;
  GenericComputeClient(hpx::id_type id) : base_type(std::move(id)) {}

  hpx::future<OpaqueOutputData> execute_task(const OpaqueInputData &inputs) {
    typedef GenericComputeServer::execute_task_action action_type;
    return hpx::async<action_type>(this->get_id(), inputs);
  }
};

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif
