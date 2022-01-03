#ifndef END_TO_END_JIT_TEST_H
#define END_TO_END_JIT_TEST_H

#include <gtest/gtest.h>

#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/JitCompilerEngine.h"
#include "llvm/Support/Path.h"

#include "globals.h"

#define ASSERT_LLVM_ERROR(err)                                                 \
  if (err) {                                                                   \
    llvm::errs() << "error: " << std::move(err) << "\n";                       \
    ASSERT_TRUE(false);                                                        \
  }

// Checks that the value `val` is not in an error state. Returns
// `true` if the test passes, otherwise `false`.
template <typename T>
static bool assert_expected_success(llvm::Expected<T> &val) {
  if (!((bool)val)) {
    llvm::errs() << llvm::toString(std::move(val.takeError())) << "\n";
    return false;
  }

  return true;
}

// Checks that the value `val` is not in an error state. Returns
// `true` if the test passes, otherwise `false`.
template <typename T>
static bool assert_expected_success(llvm::Expected<T> &&val) {
  return assert_expected_success(val);
}

// Checks that the value `val` is not in an error state. Returns
// `true` if the test passes, otherwise `false`.
template <typename T>
static bool assert_expected_failure(llvm::Expected<T> &&val) {
  return !assert_expected_success(val);
}

// Checks that the value `val` of type `llvm::Expected<T>` is not in
// an error state.
#define ASSERT_EXPECTED_SUCCESS(val)                                           \
  do {                                                                         \
    if (!assert_expected_success(val))                                         \
      GTEST_FATAL_FAILURE_("Expected<T> in error state");                      \
  } while (0)

// Checks that the value `val` of type `llvm::Expected<T>` is in
// an error state.
#define ASSERT_EXPECTED_FAILURE(val)                                           \
  do {                                                                         \
    if (assert_expected_success(val))                                          \
      GTEST_FATAL_FAILURE_("Expected<T> not in error state");                  \
  } while (0)

// Checks that the value `val` is not in an error state and is equal
// to the value given in `exp`. Returns `true` if the test passes,
// otherwise `false`.
template <typename T, typename V>
static bool assert_expected_value(llvm::Expected<T> &val, const V &exp) {
  if (!assert_expected_success(val))
    return false;

  if (!(val.get() == static_cast<T>(exp))) {
    llvm::errs() << "Expected value " << exp << ", but got " << val.get()
                 << "\n";
    return false;
  }

  return true;
}

// Checks that the value `val` is not in an error state and is equal
// to the value given in `exp`. Returns `true` if the test passes,
// otherwise `false`.
template <typename T, typename V>
static bool assert_expected_value(llvm::Expected<T> &&val, const V &exp) {
  return assert_expected_value(val, exp);
}

// Checks that the value `val` of type `llvm::Expected<T>` is not in
// an error state and is equal to the value of type `T` given in
// `exp`.
#define ASSERT_EXPECTED_VALUE(val, exp)                                        \
  do {                                                                         \
    if (!assert_expected_value(val, exp)) {                                    \
      GTEST_FATAL_FAILURE_("Expected<T> with wrong value");                    \
    }                                                                          \
  } while (0)

#define ASSERT_EQ_OUTCOME(val, exp)                                            \
  if(!val.has_value()) {                                                       \
    std::string msg = "ERROR: <" + val.error().mesg + "> \n";                  \
    GTEST_FATAL_FAILURE_(msg.c_str());                                         \
  };                                                                           \
  ASSERT_EQ(val.value(), exp);

static inline llvm::Optional<mlir::concretelang::KeySetCache> getTestKeySetCache() {

  llvm::SmallString<0> cachePath;
  llvm::sys::path::system_temp_directory(true, cachePath);
  llvm::sys::path::append(cachePath, "KeySetCache");

  auto cachePathStr = std::string(cachePath);
  return llvm::Optional<mlir::concretelang::KeySetCache>(
      mlir::concretelang::KeySetCache(cachePathStr));
}

static inline std::shared_ptr<mlir::concretelang::KeySetCache> getTestKeySetCachePtr() {
  return std::make_shared<mlir::concretelang::KeySetCache>(
    getTestKeySetCache().getValue());
}

// Jit-compiles the function specified by `func` from `src` and
// returns the corresponding lambda. Any compilation errors are caught
// and reult in abnormal termination.
template <typename F>
mlir::concretelang::JitCompilerEngine::Lambda
internalCheckedJit(F checkFunc, llvm::StringRef src,
                   llvm::StringRef func = "main",
                   bool useDefaultFHEConstraints = false,
		   bool autoParallelize = false) {

  mlir::concretelang::JitCompilerEngine engine;

  if (useDefaultFHEConstraints)
    engine.setFHEConstraints(defaultV0Constraints);
#ifdef CONCRETELANG_PARALLEL_TESTING_ENABLED
  engine.setAutoParallelize(true);
#else
  engine.setAutoParallelize(autoParallelize);
#endif

  llvm::Expected<mlir::concretelang::JitCompilerEngine::Lambda> lambdaOrErr =
      engine.buildLambda(src, func, getTestKeySetCache());

  if (!lambdaOrErr) {
    std::cout << llvm::toString(lambdaOrErr.takeError()) << std::endl;
  }
  checkFunc(lambdaOrErr);

  return std::move(*lambdaOrErr);
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
#define checkedJit(...)                                                        \
  internalCheckedJit(                                                          \
      [](llvm::Expected<mlir::concretelang::JitCompilerEngine::Lambda> &lambda) {  \
        ASSERT_EXPECTED_SUCCESS(lambda);                                       \
      },                                                                       \
      __VA_ARGS__)

#endif
