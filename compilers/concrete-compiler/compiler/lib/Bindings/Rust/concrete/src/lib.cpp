// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "lib.h"
#include "concrete-optimizer.hpp"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/V0Parameters.h"
#include "cxx.h"
#include <memory>

using mlir::concretelang::CompilationContext;
using mlir::concretelang::CompilerEngine;

namespace concrete_rust {

// CompilationOptions
std::unique_ptr<CompilationOptions> compilation_options_new() {
  return std::make_unique<CompilationOptions>(
      CompilationOptions{mlir::concretelang::CompilationOptions()});
}

void CompilationOptions::set_display_optimizer_choice(bool val) {
  this->inner.optimizerConfig.display = val;
}

void CompilationOptions::add_composition_rule(rust::Str from_func,
                                              rust::usize from_pos,
                                              rust::Str to_func,
                                              rust::usize to_pos) {
  this->inner.optimizerConfig.composition_rules.push_back(
      mlir::concretelang::optimizer::CompositionRule{
          std::string(from_func), from_pos, std::string(to_func), to_pos});
}

void CompilationOptions::set_simulate(bool val) { this->inner.simulate = val; }

void CompilationOptions::set_loop_parallelize(bool val) {
  this->inner.loopParallelize = val;
}

void CompilationOptions::set_dataflow_parallelize(bool val) {
  this->inner.dataflowParallelize = val;
}

void CompilationOptions::set_auto_parallelize(bool val) {
  this->inner.autoParallelize = val;
}

void CompilationOptions::set_compress_evaluation_keys(bool val) {
  this->inner.compressEvaluationKeys = val;
}

void CompilationOptions::set_compress_input_ciphertexts(bool val) {
  this->inner.compressInputCiphertexts = val;
}

void CompilationOptions::set_p_error(double val) {
  this->inner.optimizerConfig.p_error = val;
}

void CompilationOptions::set_global_p_error(double val) {
  this->inner.optimizerConfig.global_p_error = val;
}

void CompilationOptions::set_optimizer_strategy(uint8_t val) {
  this->inner.optimizerConfig.strategy =
      static_cast<mlir::concretelang::optimizer::Strategy>(val);
}

void CompilationOptions::set_optimizer_multi_parameter_strategy(uint8_t val) {
  this->inner.optimizerConfig.multi_param_strategy =
      static_cast<concrete_optimizer::MultiParamStrategy>(val);
}

void CompilationOptions::set_enable_tlu_fusing(bool val) {
  this->inner.enableTluFusing = val;
}

void CompilationOptions::set_print_tlu_fusing(bool val) {
  this->inner.printTluFusing = val;
}

void CompilationOptions::set_enable_overflow_detection_in_simulation(bool val) {
  this->inner.enableOverflowDetectionInSimulation = val;
}

void CompilationOptions::set_composable(bool val) {
  this->inner.optimizerConfig.composable = val;
}

void CompilationOptions::set_range_restriction(rust::Str json) {
  if (!json.empty()) {
    this->inner.optimizerConfig.range_restriction =
        std::make_shared<concrete_optimizer::restriction::RangeRestriction>(
            concrete_optimizer::restriction::range_restriction_from_json(json));
  }
}

void CompilationOptions::set_keyset_restriction(rust::Str json) {
  if (!json.empty()) {
    this->inner.optimizerConfig.keyset_restriction =
        std::make_shared<concrete_optimizer::restriction::KeysetRestriction>(
            concrete_optimizer::restriction::keyset_restriction_from_json(
                json));
  }
}

void CompilationOptions::set_security_level(uint64_t val) {
  this->inner.optimizerConfig.security = val;
}

// Library
std::unique_ptr<ProgramInfo> Library::getProgramInfo() {
  auto maybeProgramInfo = this->inner.getProgramInfo();
  if (maybeProgramInfo.has_error()) {
    throw std::runtime_error(std::move(maybeProgramInfo.error().mesg));
  }
  return std::make_unique<ProgramInfo>(ProgramInfo{maybeProgramInfo.value()});
}

rust::String Library::getStaticLibraryPath() {
  return this->inner.getStaticLibraryPath();
}

std::unique_ptr<Library> compile(rust::Str sources,
                                 const CompilationOptions &options,
                                 rust::Str outputDirPath) {
  auto cxxSources = (std::string)sources;
  auto cxxOutputDirPath = (std::string)outputDirPath;
  auto context = CompilationContext::createShared();
  auto engine = CompilerEngine(context);
  engine.setCompilationOptions(options.inner);
  std::unique_ptr<llvm::MemoryBuffer> mb =
      llvm::MemoryBuffer::getMemBuffer(cxxSources);
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
  auto maybeLibrary =
      engine.compile(sm, cxxOutputDirPath, "", false, true, true, true);
  if (auto err = maybeLibrary.takeError()) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }
  mlir::concretelang::Library output = *maybeLibrary;
  return std::make_unique<Library>(Library{output});
}

} // namespace concrete_rust
