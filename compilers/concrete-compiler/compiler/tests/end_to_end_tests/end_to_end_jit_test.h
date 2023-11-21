#ifndef END_TO_END_JIT_TEST_H
#define END_TO_END_JIT_TEST_H

#include "../tests_tools/keySetCache.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/TestLib/TestCircuit.h"
#include "cstdlib"
#include "end_to_end_test.h"
#include "globals.h"
#include "tests_tools/assert.h"
#include <gtest/gtest.h>

using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::testlib::TestCircuit;

llvm::StringRef DEFAULT_func = "main";
bool DEFAULT_useDefaultFHEConstraints = false;
bool DEFAULT_dataflowParallelize = false;
bool DEFAULT_loopParallelize = false;
bool DEFAULT_batchTFHEOps = false;
double DEFAULT_global_p_error = TEST_ERROR_RATE;
bool DEFAULT_chunkedIntegers = false;
unsigned int DEFAULT_chunkSize = 4;
unsigned int DEFAULT_chunkWidth = 2;

// Jit-compiles the function specified by `func` from `src` and
// returns the corresponding lambda. Any compilation errors are caught
// and reult in abnormal termination.
inline Result<TestCircuit> internalCheckedJit(
    llvm::StringRef src, llvm::StringRef func = DEFAULT_func,
    bool useDefaultFHEConstraints = DEFAULT_useDefaultFHEConstraints,
    bool dataflowParallelize = DEFAULT_dataflowParallelize,
    bool loopParallelize = DEFAULT_loopParallelize,
    bool batchTFHEOps = DEFAULT_batchTFHEOps,
    double global_p_error = DEFAULT_global_p_error,
    bool chunkedIntegers = DEFAULT_chunkedIntegers,
    unsigned int chunkSize = DEFAULT_chunkSize,
    unsigned int chunkWidth = DEFAULT_chunkWidth) {

  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};
  auto options =
      mlir::concretelang::CompilationOptions(std::string(func.data()));
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
#ifdef CONCRETELANG_DATAFLOW_TESTING_ENABLED
  options.dataflowParallelize = true;
  options.loopParallelize = true;
#else
  options.dataflowParallelize = dataflowParallelize;
#endif
#endif
  options.batchTFHEOps = batchTFHEOps;

  ce.setCompilationOptions(options);
  std::vector<std::string> sources = {src.str()};
  auto artifactFolder = concretelang::testlib::createTempFolderIn(
      concretelang::testlib::getSystemTempFolderPath());
  auto result = ce.compile(sources, artifactFolder);
  if (!result) {
    llvm::errs() << result.takeError();
    return StringError("Failed to compile sources....");
  }
  auto compiled = result.get();

  auto keyset =
      getTestKeySetCachePtr()
          ->getKeyset(compiled.getProgramInfo().asReader().getKeyset(), 0, 0)
          .value();
  return TestCircuit::create(
      keyset, compiled.getProgramInfo().asReader(),
      compiled.getSharedLibraryPath(compiled.getOutputDirPath()), 0, 0, false);
}

// Wrapper around `internalCheckedJit` that causes
// `ASSERT_EXPECTED_SUCCESS` to use the file and line number of the
// caller instead of `internalCheckedJit`.
#define checkedJit(VARNAME, ...)                                               \
  auto VARNAMEOrErr = internalCheckedJit(__VA_ARGS__);                         \
  ASSERT_OUTCOME_HAS_VALUE(VARNAMEOrErr);                                      \
  auto VARNAME = std::move(VARNAMEOrErr.value());

#endif
