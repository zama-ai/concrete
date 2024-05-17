#ifndef END_TO_END_JIT_TEST_H
#define END_TO_END_JIT_TEST_H

#include "../tests_tools/keySetCache.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/TestLib/TestProgram.h"
#include "cstdlib"
#include "end_to_end_test.h"
#include "globals.h"
#include "tests_tools/assert.h"
#include <gtest/gtest.h>

using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::testlib::TestProgram;

llvm::StringRef DEFAULT_func = "main";
bool DEFAULT_useDefaultFHEConstraints = false;
bool DEFAULT_dataflowParallelize = false;
bool DEFAULT_loopParallelize = false;
bool DEFAULT_batchTFHEOps = false;
double DEFAULT_global_p_error = TEST_ERROR_RATE;
bool DEFAULT_chunkedIntegers = false;
unsigned int DEFAULT_chunkSize = 4;
unsigned int DEFAULT_chunkWidth = 2;
bool DEFAULT_use_multi_parameter = true;

// Jit-compiles the function specified by `func` from `src` and
// returns the corresponding lambda. Any compilation errors are caught
// and reult in abnormal termination.
inline Result<TestProgram> internalCheckedJit(
    llvm::StringRef src, llvm::StringRef func = DEFAULT_func,
    bool useDefaultFHEConstraints = DEFAULT_useDefaultFHEConstraints,
    bool dataflowParallelize = DEFAULT_dataflowParallelize,
    bool loopParallelize = DEFAULT_loopParallelize,
    bool batchTFHEOps = DEFAULT_batchTFHEOps,
    double global_p_error = DEFAULT_global_p_error,
    bool chunkedIntegers = DEFAULT_chunkedIntegers,
    unsigned int chunkSize = DEFAULT_chunkSize,
    unsigned int chunkWidth = DEFAULT_chunkWidth,
    bool use_multi_parameter = DEFAULT_use_multi_parameter) {

  auto options = mlir::concretelang::CompilationOptions();
  options.optimizerConfig.global_p_error = global_p_error;
  options.chunkIntegers = chunkedIntegers;
  options.chunkSize = chunkSize;
  options.chunkWidth = chunkWidth;
  if (useDefaultFHEConstraints) {
    options.v0FHEConstraints = defaultV0Constraints;
    options.optimizerConfig.strategy =
        mlir::concretelang::optimizer::Strategy::V0;
  }
  options.loopParallelize = loopParallelize;
#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
  options.dataflowParallelize = dataflowParallelize;
#endif
  options.batchTFHEOps = batchTFHEOps;

  if (!use_multi_parameter)
    options.optimizerConfig.strategy =
        mlir::concretelang::optimizer::Strategy::DAG_MONO;
  std::vector<std::string> sources = {src.str()};
  TestProgram testProgram(options);
  OUTCOME_TRYV(testProgram.compile({src.str()}));
  OUTCOME_TRYV(testProgram.generateKeyset());
  return std::move(testProgram);
}

// Wrapper around `internalCheckedJit` that causes
// `ASSERT_EXPECTED_SUCCESS` to use the file and line number of the
// caller instead of `internalCheckedJit`.
#define checkedJit(VARNAME, ...)                                               \
  ASSERT_ASSIGN_OUTCOME_VALUE(VARNAME, internalCheckedJit(__VA_ARGS__));

#endif
