#ifndef END_TO_END_JIT_TEST_H
#define END_TO_END_JIT_TEST_H

#include <gtest/gtest.h>

#include "zamalang/Support/CompilerEngine.h"

mlir::zamalang::V0FHEConstraint defaultV0Constraints();

#define ASSERT_LLVM_ERROR(err)                                                 \
  if (err) {                                                                   \
    llvm::errs() << "error: " << err << "\n";                                  \
    ASSERT_TRUE(false);                                                        \
  }

#endif