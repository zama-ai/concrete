// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_DFR_DISTRIBUTED_GENERIC_TASK_SERVER_HPP
#define ZAMALANG_DFR_DISTRIBUTED_GENERIC_TASK_SERVER_HPP

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

#include "zamalang/Runtime/DFRuntime.hpp"
#include "zamalang/Runtime/key_manager.hpp"

extern WorkFunctionRegistry *node_level_work_function_registry;
extern std::list<void *> new_allocated;

using namespace hpx::naming;
using namespace hpx::components;
using namespace hpx::collectives;

struct OpaqueInputData {
  OpaqueInputData() = default;

  OpaqueInputData(std::string wfn_name, std::vector<void *> params,
                  std::vector<size_t> param_sizes,
                  std::vector<size_t> output_sizes, bool alloc_p = false)
      : wfn_name(wfn_name), params(std::move(params)),
        param_sizes(std::move(param_sizes)),
        output_sizes(std::move(output_sizes)), alloc_p(alloc_p) {}

  OpaqueInputData(const OpaqueInputData &oid)
      : wfn_name(std::move(oid.wfn_name)), params(std::move(oid.params)),
        param_sizes(std::move(oid.param_sizes)),
        output_sizes(std::move(oid.output_sizes)), alloc_p(oid.alloc_p) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar &wfn_name;
    ar &param_sizes;
    ar &output_sizes;
    for (auto p : param_sizes) {
      char *param = new char[p];
      // TODO: Optimise these serialisation operations
      for (size_t i = 0; i < p; ++i)
        ar &param[i];
      params.push_back((void *)param);
    }
    alloc_p = true;
  }
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    ar &wfn_name;
    ar &param_sizes;
    ar &output_sizes;
    for (size_t p = 0; p < params.size(); ++p)
      for (size_t i = 0; i < param_sizes[p]; ++i)
        ar &static_cast<char *>(params[p])[i];
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

  std::string wfn_name;
  std::vector<void *> params;
  std::vector<size_t> param_sizes;
  std::vector<size_t> output_sizes;
  bool alloc_p = false;
};

struct OpaqueOutputData {
  OpaqueOutputData() = default;
  OpaqueOutputData(std::vector<void *> outputs,
                   std::vector<size_t> output_sizes, bool alloc_p = false)
      : outputs(std::move(outputs)), output_sizes(std::move(output_sizes)),
        alloc_p(alloc_p) {}
  OpaqueOutputData(const OpaqueOutputData &ood)
      : outputs(std::move(ood.outputs)),
        output_sizes(std::move(ood.output_sizes)), alloc_p(ood.alloc_p) {}

  friend class hpx::serialization::access;
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar &output_sizes;
    for (auto p : output_sizes) {
      char *output = new char[p];
      for (size_t i = 0; i < p; ++i)
        ar &output[i];
      outputs.push_back((void *)output);
      new_allocated.push_back((void *)output);
    }
    alloc_p = true;
  }
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    ar &output_sizes;
    for (size_t p = 0; p < outputs.size(); ++p) {
      for (size_t i = 0; i < output_sizes[p]; ++i)
        ar &static_cast<char *>(outputs[p])[i];
      // TODO: investigate if HPX is automatically deallocating
      // these. Here it could be safely assumed that these would no
      // longer be live.
      // delete (char*)outputs[p];
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()

  std::vector<void *> outputs;
  std::vector<size_t> output_sizes;
  bool alloc_p = false;
};

struct GenericComputeServer : component_base<GenericComputeServer> {
  GenericComputeServer() = default;

  // Component actions exposed
  OpaqueOutputData execute_task(const OpaqueInputData &inputs) {
    auto wfn = node_level_work_function_registry->getWorkFunctionPointer(
        inputs.wfn_name);
    std::vector<void *> outputs;

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

    if (inputs.alloc_p)
      for (auto p : inputs.params)
        delete ((char *)p);

    return OpaqueOutputData(std::move(outputs), std::move(inputs.output_sizes),
                            inputs.alloc_p);
  }

  HPX_DEFINE_COMPONENT_ACTION(GenericComputeServer, execute_task);
};

HPX_REGISTER_ACTION_DECLARATION(GenericComputeServer::execute_task_action,
                                GenericComputeServer_execute_task_action)

HPX_REGISTER_COMPONENT_MODULE()
HPX_REGISTER_COMPONENT(hpx::components::component<GenericComputeServer>,
                       GenericComputeServer)

HPX_REGISTER_ACTION(GenericComputeServer::execute_task_action,
                    GenericComputeServer_execute_task_action)

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

#endif
