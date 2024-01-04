#ifndef UINT_TESTS_COMMON_ASSERT_H
#define UINT_TESTS_COMMON_ASSERT_H

#include "llvm/ADT/StringExtras.h"
#include <concretelang/Runtime/DFRuntime.hpp>
#include <gtest/gtest.h>

#define ASSERT_LLVM_ERROR(err)                                                 \
  {                                                                            \
    llvm::Error e = err;                                                       \
    if (e) {                                                                   \
      handleAllErrors(std::move(e), [](const llvm::ErrorInfoBase &ei) {        \
        ASSERT_TRUE(false) << ei.message();                                    \
      });                                                                      \
    }                                                                          \
  }

#define DISCARD_LLVM_ERROR(err)                                                \
  {                                                                            \
    llvm::Error e = std::move(err);                                            \
    if (e) {                                                                   \
      handleAllErrors(std::move(e), [](const llvm::ErrorInfoBase &ei) {        \
        ASSERT_TRUE(true);                                                     \
      });                                                                      \
    }                                                                          \
  }

// Checks that the value `val` is not in an error state. Returns
// `true` if the test passes, otherwise `false`.
template <typename T>
static bool assert_expected_success(llvm::Expected<T> &val) {
  if (!((bool)val)) {
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
  if (!((bool)val)) {
    if (!mlir::concretelang::dfr::_dfr_is_root_node()) {
      llvm::toString(val.takeError());
      return true;
    }
    // We need to consume the error, so let's do it here
    llvm::errs() << "assert_expected_failure: "
                 << llvm::toString(val.takeError()) << "\n";
    return true;
  }
  return false;
}

// Checks that the value `val` of type `llvm::Expected<T>` is not in
// an error state.
#define ASSERT_EXPECTED_SUCCESS(val)                                           \
  do {                                                                         \
    if (!assert_expected_success(val)) {                                       \
      GTEST_FATAL_FAILURE_("Expected<T> in error state")                       \
          << llvm::toString(val.takeError());                                  \
    }                                                                          \
  } while (0)

// Checks that the value `val` of type `llvm::Expected<T>` is in
// an error state.
#define ASSERT_EXPECTED_FAILURE(val)                                           \
  do {                                                                         \
    if (!assert_expected_failure(val))                                         \
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
      GTEST_FATAL_FAILURE_("Expected<T> with wrong value")                     \
          << llvm::toString(val.takeError());                                  \
    }                                                                          \
  } while (0)

#define ASSERT_EQ_OUTCOME(val, exp)                                            \
  if (!val.has_value()) {                                                      \
    std::string msg = "ERROR: <" + val.error().mesg + "> \n";                  \
    GTEST_FATAL_FAILURE_(msg.c_str());                                         \
  };                                                                           \
  ASSERT_EQ(val.value(), exp);

#define ASSERT_ASSIGN_OUTCOME_VALUE(ident, val)                                \
  auto ident__ = val;                                                          \
  if (!ident__.has_value()) {                                                  \
    std::string msg = "Outcome failure " + ident__.error().mesg;               \
    GTEST_FATAL_FAILURE_(msg.c_str());                                         \
  }                                                                            \
  auto ident = std::move(ident__.value());

#define ASSERT_OUTCOME_HAS_VALUE(val)                                          \
  {                                                                            \
    auto tmp = val;                                                            \
    if (!tmp.has_value()) {                                                    \
      std::string msg = "Outcome failure " + tmp.error().mesg;                 \
      GTEST_FATAL_FAILURE_(msg.c_str());                                       \
    }                                                                          \
  }

#define ASSERT_OUTCOME_HAS_FAILURE(val)                                        \
  {                                                                            \
    auto tmp = val;                                                            \
    if (tmp.has_value()) {                                                     \
      GTEST_FATAL_FAILURE_("Outcome value when failure expected");             \
    }                                                                          \
  }

#define ASSERT_OUTCOME_HAS_FAILURE_WITH_ERRORMSG(val, errmsg)                  \
  {                                                                            \
    auto tmp = val;                                                            \
    if (tmp.has_value()) {                                                     \
      GTEST_FATAL_FAILURE_("Outcome value when failure expected");             \
    }                                                                          \
    ASSERT_EQ(tmp.error().mesg, errmsg);                                       \
  }

#endif
