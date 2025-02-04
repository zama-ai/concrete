// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

/// This file contains entry points needed when linking ConcretelangSupport.
/// This allows to compile ConcreteSys without link with the actual Runtime, OMP and cuda stuffs.
/// To be revisited if DFR or GPU is needed in precompiles.
namespace mlir {
namespace concretelang {
namespace dfr {
    void _dfr_set_use_omp(bool use_omp) { }
    bool _dfr_is_root_node() { return true; }
    void _dfr_set_required(bool is_required) {}
} // namespace dfr
namespace gpu_dfg {
    bool check_cuda_device_available() {return false;}
    bool check_cuda_runtime_enabled() {return false;}
} // namespace gpu_dfg
} // namespace concretelang
} // namespace mlir
