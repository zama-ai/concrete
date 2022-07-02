#ifndef END_TO_END_JIT_TEST_H
#define END_TO_END_JIT_TEST_H

#include <gtest/gtest.h>

#include "../tests_tools/keySetCache.h"

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/JITSupport.h"

#include "globals.h"
#include "tests_tools/assert.h"

// Jit-compiles the function specified by `func` from `src` and
// returns the corresponding lambda. Any compilation errors are caught
// and reult in abnormal termination.
inline llvm::Expected<
    mlir::concretelang::ClientServer<mlir::concretelang::JITSupport>>
internalCheckedJit(llvm::StringRef src, llvm::StringRef func = "main",
                   bool useDefaultFHEConstraints = false,
                   bool dataflowParallelize = false,
                   bool loopParallelize = false) {

  auto options =
      mlir::concretelang::CompilationOptions(std::string(func.data()));
  if (useDefaultFHEConstraints)
    options.v0FHEConstraints = defaultV0Constraints;

  // Allow loop parallelism in all cases
  options.loopParallelize = loopParallelize;
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
#ifdef CONCRETELANG_PARALLEL_TESTING_ENABLED
  options.dataflowParallelize = true;
  options.loopParallelize = true;
#else
  options.dataflowParallelize = dataflowParallelize;
#endif
#endif

  auto lambdaOrErr =
      mlir::concretelang::ClientServer<mlir::concretelang::JITSupport>::create(
          src, options, getTestKeySetCache(), mlir::concretelang::JITSupport());

  return lambdaOrErr;
}

// Shorthands to create integer literals of a specific type
static inline uint8_t operator"" _u8(unsigned long long int v) { return v; }
static inline uint16_t operator"" _u16(unsigned long long int v) { return v; }
static inline uint32_t operator"" _u32(unsigned long long int v) { return v; }
static inline uint64_t operator"" _u64(unsigned long long int v) { return v; }

// Evaluates to the number of elements of a statically initialized
// array
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

// Wrapper around `internalCheckedJit` that causes
// `ASSERT_EXPECTED_SUCCESS` to use the file and line number of the
// caller instead of `internalCheckedJit`.
#define checkedJit(VARNAME, ...)                                               \
  auto VARNAMEOrErr = internalCheckedJit(__VA_ARGS__);                         \
  ASSERT_EXPECTED_SUCCESS(VARNAMEOrErr);                                       \
  auto VARNAME = std::move(*VARNAMEOrErr);

#endif
