// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <dlfcn.h>

#include "boost/outcome.h"

#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Common/Error.h"
#include "concretelang/ServerLib/DynamicArityCall.h"
#include "concretelang/ServerLib/DynamicModule.h"
#include "concretelang/ServerLib/DynamicRankCall.h"
#include "concretelang/ServerLib/ServerLambda.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"

namespace concretelang {
namespace serverlib {

using concretelang::clientlib::CircuitGate;
using concretelang::clientlib::CircuitGateShape;
using concretelang::clientlib::PublicArguments;
using concretelang::error::StringError;

void next_coord_index(size_t index[], size_t sizes[], size_t rank) {
  // increase multi dim index
  for (int r = rank - 1; r >= 0; r--) {
    if (index[r] < sizes[r] - 1) {
      index[r]++;
      return;
    }
    index[r] = 0;
  }
}

size_t global_index(size_t index[], size_t sizes[], size_t strides[],
                    size_t rank) {
  // compute global index from multi dim index
  size_t g_index = 0;
  size_t default_stride = 1;
  for (int r = rank - 1; r >= 0; r--) {
    g_index += index[r] * ((strides[r] == 0) ? default_stride : strides[r]);
    default_stride *= sizes[r];
  }
  return g_index;
}

/** Helper function to convert from MemRefDescriptor to
 * encrypted_scalars_and_sizes_t assuming MemRefDescriptor are bufferized */
encrypted_scalars_and_sizes_t encrypted_scalars_and_sizes_t_from_MemRef(
    size_t memref_rank, encrypted_scalars_t allocated,
    encrypted_scalars_t aligned, size_t offset, size_t *sizes,
    size_t *strides) {
  encrypted_scalars_and_sizes_t result;
  assert(aligned != nullptr);
  result.sizes.resize(memref_rank);
  for (size_t r = 0; r < memref_rank; r++) {
    result.sizes[r] = sizes[r];
  }
  size_t *index = new size_t[memref_rank]; // ephemeral multi dim index to
                                           // compute global strides
  for (size_t r = 0; r < memref_rank; r++) {
    index[r] = 0;
  }
  auto len = result.length();
  result.values.resize(len);
  // TODO: add a fast path for dense result (no real strides)
  for (size_t i = 0; i < len; i++) {
    int g_index = offset + global_index(index, sizes, strides, memref_rank);
    result.values[i] = aligned[offset + g_index];
    next_coord_index(index, sizes, memref_rank);
  }
  delete[] index;
  // TEMPORARY: That quick and dirty but as this function is used only to
  // convert a result of the mlir program and as data are copied here, we
  // release the alocated pointer if it set.
  if (allocated != nullptr) {
    free(allocated);
  }
  return result;
}

outcome::checked<ServerLambda, StringError>
ServerLambda::loadFromModule(std::shared_ptr<DynamicModule> module,
                             std::string funcName) {
  ServerLambda lambda;
  lambda.module =
      module; // prevent module and library handler from being destroyed
  lambda.func =
      (void *(*)(void *, ...))dlsym(module->libraryHandle, funcName.c_str());

  if (auto err = dlerror()) {
    return StringError("Cannot open lambda:") << std::string(err);
  }

  auto param =
      llvm::find_if(module->clientParametersList, [&](ClientParameters param) {
        return param.functionName == funcName;
      });

  if (param == module->clientParametersList.end()) {
    return StringError("cannot find function ")
           << funcName << "in client parameters";
  }

  if (param->outputs.size() != 1) {
    return StringError("ServerLambda: output arity (")
           << std::to_string(param->outputs.size())
           << ") != 1 is not supported";
  }

  if (!param->outputs[0].encryption.hasValue()) {
    return StringError("ServerLambda: clear output is not yet supported");
  }

  lambda.clientParameters = *param;
  return lambda;
}

outcome::checked<ServerLambda, StringError>
ServerLambda::load(std::string funcName, std::string outputLib) {
  OUTCOME_TRY(auto module, DynamicModule::open(outputLib));
  return ServerLambda::loadFromModule(module, funcName);
}

encrypted_scalars_and_sizes_t dynamicCall(void *(*func)(void *...),
                                          std::vector<void *> &preparedArgs,
                                          CircuitGate &output) {
  size_t rank = output.shape.dimensions.size();
  return multi_arity_call_dynamic_rank(func, preparedArgs, rank);
}

std::unique_ptr<clientlib::PublicResult>
ServerLambda::call(PublicArguments &args) {
  std::vector<void *> preparedArgs(args.preparedArgs.begin(),
                                   args.preparedArgs.end());
  preparedArgs.push_back((void *)&args.runtimeContext);
  return clientlib::PublicResult::fromBuffers(
      clientParameters,
      {dynamicCall(this->func, preparedArgs, clientParameters.outputs[0])});
  ;
}

} // namespace serverlib
} // namespace concretelang
