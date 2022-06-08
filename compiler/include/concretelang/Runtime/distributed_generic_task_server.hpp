// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
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

#include "concretelang/ClientLib/EvaluationKeys.h"
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

static inline void _dfr_checked_aligned_alloc(void **out, size_t align,
                                              size_t size) {
  int res = posix_memalign(out, align, size);
  if (res == ENOMEM)
    HPX_THROW_EXCEPTION(hpx::no_success, "DFR: memory allocation failed",
                        "Error: insufficient memory available.");
  if (res == EINVAL)
    HPX_THROW_EXCEPTION(hpx::no_success, "DFR: memory allocation failed",
                        "Error: invalid memory alignment.");
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
        source_locality(hpx::find_here()), ksk_id(0), bsk_id(0) {}

  OpaqueInputData(const OpaqueInputData &oid)
      : wfn_name(std::move(oid.wfn_name)), params(std::move(oid.params)),
        param_sizes(std::move(oid.param_sizes)),
        param_types(std::move(oid.param_types)),
        output_sizes(std::move(oid.output_sizes)),
        output_types(std::move(oid.output_types)), alloc_p(oid.alloc_p),
        source_locality(oid.source_locality), ksk_id(oid.ksk_id),
        bsk_id(oid.bsk_id) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar >> wfn_name;
    ar >> param_sizes >> param_types;
    ar >> output_sizes >> output_types;
    ar >> source_locality;
    for (size_t p = 0; p < param_sizes.size(); ++p) {
      char *param;
      _dfr_checked_aligned_alloc((void **)&param, 64, param_sizes[p]);
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
      case _DFR_TASK_ARG_CONTEXT: {
        ar >> bsk_id >> ksk_id;

        delete ((char *)params[p]);
        // TODO: this might be relaxed with newer versions of HPX.
        // Do not set the context here as remote operations are
        // unstable when initiated within a HPX helper thread.
        params[p] =
            (void *)
                _dfr_node_level_runtime_context_manager->getContextAddress();
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
    ar << source_locality;
    for (size_t p = 0; p < params.size(); ++p) {
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
      case _DFR_TASK_ARG_CONTEXT: {
        mlir::concretelang::RuntimeContext *context =
            *static_cast<mlir::concretelang::RuntimeContext **>(params[p]);
        LweKeyswitchKey_u64 *ksk = get_keyswitch_key_u64(context);
        LweBootstrapKey_u64 *bsk = get_bootstrap_key_u64(context);

        assert(bsk != nullptr && ksk != nullptr && "Missing context keys");
        std::cout << "Registering Key ids " << (uint64_t)ksk << " "
                  << (uint64_t)bsk << "\n"
                  << std::flush;
        _dfr_register_bsk(bsk, (uint64_t)bsk);
        _dfr_register_ksk(ksk, (uint64_t)ksk);
        ar << (uint64_t)bsk << (uint64_t)ksk;
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
  uint64_t ksk_id;
  uint64_t bsk_id;
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
      char *output;
      _dfr_checked_aligned_alloc((void **)&output, 64, (output_sizes[p]));

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
        UnrankedMemRefType<char> umref = {(int64_t)rank, outputs[p]};
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

    if (inputs.source_locality != hpx::find_here() &&
        (inputs.ksk_id || inputs.bsk_id)) {
      _dfr_node_level_runtime_context_manager->getContext(
          inputs.ksk_id, inputs.bsk_id, inputs.source_locality);
    }

    _dfr_debug_print_task(inputs.wfn_name.c_str(), inputs.params.size(),
                          inputs.output_sizes.size());
    hpx::cout << std::flush;

    switch (inputs.output_sizes.size()) {
    case 1: {
      void *output;
      _dfr_checked_aligned_alloc(&output, 512, inputs.output_sizes[0]);
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
      case 17:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], output);
        break;
      case 18:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17], output);
        break;
      case 19:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17],
            inputs.params[18], output);
        break;
      case 20:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17],
            inputs.params[18], inputs.params[19], output);
        break;
      default:
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "GenericComputeServer::execute_task",
                            "Error: number of task parameters not supported.");
      }
      outputs = {output};
      break;
    }
    case 2: {
      void *output1, *output2;
      _dfr_checked_aligned_alloc(&output1, 512, inputs.output_sizes[0]);
      _dfr_checked_aligned_alloc(&output2, 512, inputs.output_sizes[1]);
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
      case 17:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], output1, output2);
        break;
      case 18:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17], output1,
            output2);
        break;
      case 19:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17],
            inputs.params[18], output1, output2);
        break;
      case 20:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17],
            inputs.params[18], inputs.params[19], output1, output2);
        break;
      default:
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "GenericComputeServer::execute_task",
                            "Error: number of task parameters not supported.");
      }
      outputs = {output1, output2};
      break;
    }
    case 3: {
      void *output1, *output2, *output3;
      _dfr_checked_aligned_alloc(&output1, 512, inputs.output_sizes[0]);
      _dfr_checked_aligned_alloc(&output2, 512, inputs.output_sizes[1]);
      _dfr_checked_aligned_alloc(&output2, 512, inputs.output_sizes[2]);
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
      case 17:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], output1, output2, output3);
        break;
      case 18:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17], output1,
            output2, output3);
        break;
      case 19:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17],
            inputs.params[18], output1, output2, output3);
        break;
      case 20:
        wfn(inputs.params[0], inputs.params[1], inputs.params[2],
            inputs.params[3], inputs.params[4], inputs.params[5],
            inputs.params[6], inputs.params[7], inputs.params[8],
            inputs.params[9], inputs.params[10], inputs.params[11],
            inputs.params[12], inputs.params[13], inputs.params[14],
            inputs.params[15], inputs.params[16], inputs.params[17],
            inputs.params[18], inputs.params[19], output1, output2, output3);
        break;
      default:
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "GenericComputeServer::execute_task",
                            "Error: number of task parameters not supported.");
      }
      outputs = {output1, output2, output3};
      break;
    }
    default:
      HPX_THROW_EXCEPTION(hpx::no_success, "GenericComputeServer::execute_task",
                          "Error: number of task outputs not supported.");
    }

    // Deallocate input data buffers from OID deserialization (load)
    if (!_dfr_is_root_node()) {
      for (size_t p = 0; p < inputs.param_sizes.size(); ++p) {
        if (_dfr_get_arg_type(inputs.param_types[p]) != _DFR_TASK_ARG_CONTEXT) {
          if (_dfr_get_arg_type(inputs.param_types[p]) == _DFR_TASK_ARG_MEMREF)
            delete (static_cast<StridedMemRefType<char, 1> *>(inputs.params[p])
                        ->data);
          delete ((char *)inputs.params[p]);
        }
      }
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
