#ifndef END_TO_END_JIT_TEST_H
#define END_TO_END_JIT_TEST_H

#include <gtest/gtest.h>

#include "../tests_tools/keySetCache.h"

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/JITSupport.h"

#include "end_to_end_test.h"
#include "globals.h"
#include "tests_tools/assert.h"

llvm::StringRef DEFAULT_func = "main";
bool DEFAULT_useDefaultFHEConstraints = false;
bool DEFAULT_dataflowParallelize = false;
bool DEFAULT_loopParallelize = false;
bool DEFAULT_batchConcreteOps = false;
double DEFAULT_global_p_error = TEST_ERROR_RATE;
bool DEFAULT_chunkedIntegers = false;
unsigned int DEFAULT_chunkSize = 4;
unsigned int DEFAULT_chunkWidth = 2;

// Jit-compiles the function specified by `func` from `src` and
// returns the corresponding lambda. Any compilation errors are caught
// and reult in abnormal termination.
inline llvm::Expected<
    mlir::concretelang::ClientServer<mlir::concretelang::JITSupport>>
internalCheckedJit(
    llvm::StringRef src, llvm::StringRef func = DEFAULT_func,
    bool useDefaultFHEConstraints = DEFAULT_useDefaultFHEConstraints,
    bool dataflowParallelize = DEFAULT_dataflowParallelize,
    bool loopParallelize = DEFAULT_loopParallelize,
    bool batchConcreteOps = DEFAULT_batchConcreteOps,
    double global_p_error = DEFAULT_global_p_error,
    bool chunkedIntegers = DEFAULT_chunkedIntegers,
    unsigned int chunkSize = DEFAULT_chunkSize,
    unsigned int chunkWidth = DEFAULT_chunkWidth) {

  auto options =
      mlir::concretelang::CompilationOptions(std::string(func.data()));
  options.optimizerConfig.global_p_error = global_p_error;
  options.chunkIntegers = chunkedIntegers;
  options.chunkSize = chunkSize;
  options.chunkWidth = chunkWidth;
  if (useDefaultFHEConstraints) {
    options.v0FHEConstraints = defaultV0Constraints;
    options.optimizerConfig.strategy_v0 = true;
  }

  // Allow loop parallelism in all cases
  options.loopParallelize = loopParallelize;
#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
#ifdef CONCRETELANG_DATAFLOW_TESTING_ENABLED
  options.dataflowParallelize = true;
  options.loopParallelize = true;
#else
  options.dataflowParallelize = dataflowParallelize;
#endif
#endif
  options.batchConcreteOps = batchConcreteOps;

  auto lambdaOrErr =
      mlir::concretelang::ClientServer<mlir::concretelang::JITSupport>::create(
          src, options, getTestKeySetCache(), mlir::concretelang::JITSupport());

  return lambdaOrErr;
}

// Wrapper around `internalCheckedJit` that causes
// `ASSERT_EXPECTED_SUCCESS` to use the file and line number of the
// caller instead of `internalCheckedJit`.
#define checkedJit(VARNAME, ...)                                               \
  auto VARNAMEOrErr = internalCheckedJit(__VA_ARGS__);                         \
  ASSERT_EXPECTED_SUCCESS(VARNAMEOrErr);                                       \
  auto VARNAME = std::move(*VARNAMEOrErr);

#endif
