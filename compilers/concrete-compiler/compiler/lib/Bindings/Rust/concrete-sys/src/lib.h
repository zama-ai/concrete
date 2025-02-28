// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETE_SYS_LIB_H
#define CONCRETE_SYS_LIB_H

#include "cxx.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/V0Parameters.h"
#include <_types/_uint8_t.h>
#include <memory>
#include <sys/_types/_int64_t.h>

// Contains helpers not available from the original compiler api.
namespace concrete_sys{

struct CompilationOptions{
    mlir::concretelang::CompilationOptions inner;
    void set_display_optimizer_choice(bool);
    void set_loop_parallelize(bool);
    void set_dataflow_parallelize(bool);
    void set_auto_parallelize(bool);
    void set_compress_evaluation_keys(bool);
    void set_compress_input_ciphertexts(bool);
    void set_p_error(double);
    void set_global_p_error(double);
    void set_optimizer_strategy(uint8_t);
    void set_optimizer_multi_parameter_strategy(uint8_t);
    void set_enable_tlu_fusing(bool);
    void set_print_tlu_fusing(bool);
    void set_enable_overflow_detection_in_simulation(bool);
    void set_simulate(bool);
    void set_composable(bool);
    void set_range_restriction(rust::Str);
    void set_keyset_restriction(rust::Str);
    void set_security_level(uint64_t);
    void add_composition_rule(rust::Str, rust::usize, rust::Str, rust::usize);
};

std::unique_ptr<CompilationOptions> compilation_options_new();

struct ProgramInfo {
    concretelang::protocol::Message<concreteprotocol::ProgramInfo> inner;
};

struct Library {
    mlir::concretelang::Library inner;
    std::unique_ptr<ProgramInfo> getProgramInfo();
    rust::String getStaticLibraryPath();
};

std::unique_ptr<Library> compile(
    rust::Str sm,
    const CompilationOptions &options,
    rust::Str outputDirPath
);

} // namespace concrete_sys

#endif
