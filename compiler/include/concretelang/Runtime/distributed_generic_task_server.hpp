// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_DISTRIBUTED_GENERIC_TASK_SERVER_HPP
#define CONCRETELANG_DFR_DISTRIBUTED_GENERIC_TASK_SERVER_HPP

#include <cstdarg>
#include <cstdlib>
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

using namespace hpx::naming;
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

struct OpaqueInputData {
  OpaqueInputData() = default;

  OpaqueInputData(std::string _wfn_name, std::vector<void *> _params,
                  std::vector<size_t> _param_sizes,
                  std::vector<uint64_t> _param_types,
                  std::vector<size_t> _output_sizes,
                  std::vector<uint64_t> _output_types, bool _alloc_p = false)
      : wfn_name(_wfn_name), params(std::move(_params)),
        param_sizes(std::move(_param_sizes)),
        param_types(std::move(_param_types)),
        output_sizes(std::move(_output_sizes)),
        output_types(std::move(_output_types)), alloc_p(_alloc_p),
        source_locality(hpx::find_here()) {}

  OpaqueInputData(const OpaqueInputData &oid)
      : wfn_name(std::move(oid.wfn_name)), params(std::move(oid.params)),
        param_sizes(std::move(oid.param_sizes)),
        param_types(std::move(oid.param_types)),
        output_sizes(std::move(oid.output_sizes)),
        output_types(std::move(oid.output_types)), alloc_p(oid.alloc_p),
        source_locality(oid.source_locality) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar >> wfn_name;
    ar >> param_sizes >> param_types;
    ar >> output_sizes >> output_types;
    for (size_t p = 0; p < param_sizes.size(); ++p) {
      char *param = new char[param_sizes[p]];
      new_allocated.push_back((void *)param);
      ar >> hpx::serialization::make_array(param, param_sizes[p]);
      params.push_back((void *)param);

      switch (_dfr_get_arg_type(param_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(param_sizes[p]);
        UnrankedMemRefType<char> umref = {rank, params[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(param_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        size_t alloc_size = (size + mref.offset) * elementSize;
        char *data = new char[alloc_size];
        new_allocated.push_back((void *)data);
        ar >> hpx::serialization::make_array(data + mref.offset * elementSize,
                                             size * elementSize);
        static_cast<StridedMemRefType<char, 1> *>(params[p])->basePtr = nullptr;
        static_cast<StridedMemRefType<char, 1> *>(params[p])->data = data;
      } break;
      case _DFR_TASK_ARG_CONTEXT: {
        uint64_t bsk_id, ksk_id;
        ar >> bsk_id >> ksk_id >> source_locality;

        mlir::concretelang::RuntimeContext *context =
            new mlir::concretelang::RuntimeContext;
        new_allocated.push_back((void *)context);
        mlir::concretelang::RuntimeContext **_context =
            new mlir::concretelang::RuntimeContext *[1];
        new_allocated.push_back((void *)_context);
        _context[0] = context;

        context->bsk = (LweBootstrapKey_u64 *)bsk_id;
        context->ksk = (LweKeyswitchKey_u64 *)ksk_id;
        params[p] = (void *)_context;
      } break;
      case _DFR_TASK_ARG_UNRANKED_MEMREF:
      default:
        HPX_THROW_EXCEPTION(hpx::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
    alloc_p = true;
  }
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    ar << wfn_name;
    ar << param_sizes << param_types;
    ar << output_sizes << output_types;
    for (size_t p = 0; p < params.size(); ++p) {
      // Save the first level of the data structure - if the parameter
      // is a tensor/memref, there is a second level.
      ar << hpx::serialization::make_array((char *)params[p], param_sizes[p]);
      switch (_dfr_get_arg_type(param_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(param_sizes[p]);
        UnrankedMemRefType<char> umref = {rank, params[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(param_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        ar << hpx::serialization::make_array(
            mref.data + mref.offset * elementSize, size * elementSize);
      } break;
      case _DFR_TASK_ARG_CONTEXT: {
        mlir::concretelang::RuntimeContext *context =
            *static_cast<mlir::concretelang::RuntimeContext **>(params[p]);
        LweKeyswitchKey_u64 *ksk = context->ksk;
        LweBootstrapKey_u64 *bsk = context->bsk;

        // TODO: find better unique identifiers.  This is not a
        // correctness issue, but performance.
        uint64_t bsk_id = (uint64_t)bsk;
        uint64_t ksk_id = (uint64_t)ksk;

        assert(bsk != nullptr && ksk != nullptr && "Missing context keys");
        _dfr_register_bsk(bsk, bsk_id);
        _dfr_register_ksk(ksk, ksk_id);
        ar << bsk_id << ksk_id << source_locality;
      } break;
      case _DFR_TASK_ARG_UNRANKED_MEMREF:
      default:
        HPX_THROW_EXCEPTION(hpx::no_success, "DFR: OpaqueInputData save",
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
  bool alloc_p = false;
  hpx::naming::id_type source_locality;
};

struct OpaqueOutputData {
  OpaqueOutputData() = default;
  OpaqueOutputData(std::vector<void *> outputs,
                   std::vector<size_t> output_sizes,
                   std::vector<uint64_t> output_types, bool alloc_p = false)
      : outputs(std::move(outputs)), output_sizes(std::move(output_sizes)),
        output_types(std::move(output_types)), alloc_p(alloc_p) {}
  OpaqueOutputData(const OpaqueOutputData &ood)
      : outputs(std::move(ood.outputs)),
        output_sizes(std::move(ood.output_sizes)),
        output_types(std::move(ood.output_types)), alloc_p(ood.alloc_p) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar >> output_sizes >> output_types;
    for (size_t p = 0; p < output_sizes.size(); ++p) {
      char *output = new char[output_sizes[p]];
      new_allocated.push_back((void *)output);
      ar >> hpx::serialization::make_array(output, output_sizes[p]);
      outputs.push_back((void *)output);

      switch (_dfr_get_arg_type(output_types[p])) {
      case _DFR_TASK_ARG_BASE:
        break;
      case _DFR_TASK_ARG_MEMREF: {
        size_t rank = _dfr_get_memref_rank(output_sizes[p]);
        UnrankedMemRefType<char> umref = {rank, outputs[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(output_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        size_t alloc_size = (size + mref.offset) * elementSize;
        char *data = new char[alloc_size];
        new_allocated.push_back((void *)data);
        ar >> hpx::serialization::make_array(data + mref.offset * elementSize,
                                             size * elementSize);
        static_cast<StridedMemRefType<char, 1> *>(outputs[p])->basePtr =
            nullptr;
        static_cast<StridedMemRefType<char, 1> *>(outputs[p])->data = data;
      } break;
      case _DFR_TASK_ARG_CONTEXT: {

      } break;
      case _DFR_TASK_ARG_UNRANKED_MEMREF:
      default:
        HPX_THROW_EXCEPTION(hpx::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
    alloc_p = true;
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
        UnrankedMemRefType<char> umref = {rank, outputs[p]};
        DynamicMemRefType<char> mref(umref);
        size_t elementSize = _dfr_get_memref_element_size(output_types[p]);
        size_t size = 1;
        for (size_t r = 0; r < rank; ++r)
          size *= mref.sizes[r];
        ar << hpx::serialization::make_array(
            mref.data + mref.offset * elementSize, size * elementSize);
      } break;
      case _DFR_TASK_ARG_CONTEXT: {

      } break;
      case _DFR_TASK_ARG_UNRANKED_MEMREF:
      default:
        HPX_THROW_EXCEPTION(hpx::no_success, "DFR: OpaqueInputData save",
                            "Error: invalid task argument type.");
      }
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

  std::vector<void *> outputs;
  std::vector<size_t> output_sizes;
  std::vector<uint64_t> output_types;
  bool alloc_p = false;
};

struct GenericComputeServer : component_base<GenericComputeServer> {
  GenericComputeServer() = default;

  // Component actions exposed
  OpaqueOutputData execute_task(const OpaqueInputData &inputs) {
    auto wfn = _dfr_node_level_work_function_registry->getWorkFunctionPointer(
        inputs.wfn_name);
    std::vector<void *> outputs;

    if (!_dfr_is_root_node()) {
      for (size_t p = 0; p < inputs.params.size(); ++p) {
        if (_dfr_get_arg_type(inputs.param_types[p] == _DFR_TASK_ARG_CONTEXT)) {
          mlir::concretelang::RuntimeContext *ctx =
              (*(mlir::concretelang::RuntimeContext **)inputs.params[p]);
          ctx->ksk = _dfr_get_ksk(inputs.source_locality, (uint64_t)ctx->ksk);
          ctx->bsk = _dfr_get_bsk(inputs.source_locality, (uint64_t)ctx->bsk);
          break;
        }
      }
    }
    _dfr_debug_print_task(inputs.wfn_name.c_str(), inputs.params.size(),
                          inputs.output_sizes.size());
    hpx::cout << std::flush;
    switch (inputs.output_sizes.size()) {
    case 1: {
      void *output = (void *)(new char[inputs.output_sizes[0]]);
      switch (inputs.params.size()) {
      case 0:
        wfn(output);
        break;
      case 1:
        wfn(inputs.params[0], output);
        break;
      case 2:
        wfn(inputs.params[0], inputs.params[1], output);
        break;
      case 3:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2], output);
        break;
      case 4:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], output);
        break;
      case 5:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], output);
        break;
      case 6:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5], output);
        break;
      case 7:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], output);
        break;
      case 8:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], output);
        break;
      case 9:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8], output);
        break;
      case 10:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], output);
        break;
      case 11:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], output);
        break;
      case 12:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11], output);
        break;
      case 13:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], output);
        break;
      case 14:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], output);
        break;
      case 15:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14], output);
        break;
      case 16:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], output);
        break;
      default:
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "GenericComputeServer::execute_task",
                            "Error: number of task parameters not supported.");
      }
      outputs = {output};
      new_allocated.push_back(output);
      break;
    }
    case 2: {
      void *output1 = (void *)(new char[inputs.output_sizes[0]]);
      void *output2 = (void *)(new char[inputs.output_sizes[1]]);
      switch (inputs.params.size()) {
      case 0:
        wfn(output1, output2);
        break;
      case 1:
        wfn(inputs.params[0], output1, output2);
        break;
      case 2:
        wfn(inputs.params[0], inputs.params[1], output1, output2);
        break;
      case 3:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2], output1,
            output2);
        break;
      case 4:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], output1, output2);
        break;
      case 5:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], output1, output2);
        break;
      case 6:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5], output1,
            output2);
        break;
      case 7:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], output1, output2);
        break;
      case 8:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], output1, output2);
        break;
      case 9:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8], output1,
            output2);
        break;
      case 10:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], output1, output2);
        break;
      case 11:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], output1, output2);
        break;
      case 12:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11], output1,
            output2);
        break;
      case 13:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], output1, output2);
        break;
      case 14:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], output1, output2);
        break;
      case 15:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14], output1,
            output2);
        break;
      case 16:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], output1, output2);
        break;
      default:
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "GenericComputeServer::execute_task",
                            "Error: number of task parameters not supported.");
      }
      outputs = {output1, output2};
      new_allocated.push_back(output1);
      new_allocated.push_back(output2);
      break;
    }
    case 3: {
      void *output1 = (void *)(new char[inputs.output_sizes[0]]);
      void *output2 = (void *)(new char[inputs.output_sizes[1]]);
      void *output3 = (void *)(new char[inputs.output_sizes[2]]);
      switch (inputs.params.size()) {
      case 0:
        wfn(output1, output2, output3);
        break;
      case 1:
        wfn(inputs.params[0], output1, output2, output3);
        break;
      case 2:
        wfn(inputs.params[0], inputs.params[1], output1, output2, output3);
        break;
      case 3:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2], output1,
            output2, output3);
        break;
      case 4:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], output1, output2, output3);
        break;
      case 5:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], output1, output2, output3);
        break;
      case 6:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5], output1,
            output2, output3);
        break;
      case 7:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], output1, output2, output3);
        break;
      case 8:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], output1, output2, output3);
        break;
      case 9:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8], output1,
            output2, output3);
        break;
      case 10:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], output1, output2, output3);
        break;
      case 11:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], output1, output2, output3);
        break;
      case 12:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11], output1,
            output2, output3);
        break;
      case 13:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], output1, output2, output3);
        break;
      case 14:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], output1, output2, output3);
        break;
      case 15:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14], output1,
            output2, output3);
        break;
      case 16:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], output1, output2, output3);
        break;
      default:
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "GenericComputeServer::execute_task",
                            "Error: number of task parameters not supported.");
      }
      outputs = {output1, output2, output3};
      new_allocated.push_back(output1);
      new_allocated.push_back(output2);
      new_allocated.push_back(output3);
      break;
    }
    default:
      HPX_THROW_EXCEPTION(hpx::no_success, "GenericComputeServer::execute_task",
                          "Error: number of task outputs not supported.");
    }

    return OpaqueOutputData(std::move(outputs), std::move(inputs.output_sizes),
                            std::move(inputs.output_types), inputs.alloc_p);
  }

  HPX_DEFINE_COMPONENT_ACTION(GenericComputeServer, execute_task);
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
  GenericComputeClient(id_type id) : base_type(std::move(id)) {}

  hpx::future<OpaqueOutputData> execute_task(const OpaqueInputData &inputs) {
    typedef GenericComputeServer::execute_task_action action_type;
    return hpx::async<action_type>(this->get_id(), inputs);
  }
};

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif
