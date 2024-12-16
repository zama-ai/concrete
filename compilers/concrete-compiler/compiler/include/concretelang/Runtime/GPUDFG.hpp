// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_GPUDFG_HPP
#define CONCRETELANG_GPUDFG_HPP

#ifdef CONCRETELANG_CUDA_SUPPORT
#include "device.h"
#include "keyswitch.h"
#include "linear_algebra.h"
#include "programmable_bootstrap.h"

#endif

namespace mlir {
namespace concretelang {
namespace gpu_dfg {

bool check_cuda_device_available();
bool check_cuda_runtime_enabled();

} // namespace gpu_dfg
} // namespace concretelang
} // namespace mlir

#endif
