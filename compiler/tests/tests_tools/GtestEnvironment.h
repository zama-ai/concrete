#ifndef GTEST_ENVIRONMENT_H
#define GTEST_ENVIRONMENT_H

#include "concretelang/Runtime/DFRuntime.hpp"
#include <gtest/gtest.h>

class DFREnvironment : public ::testing::Environment {
public:
  ~DFREnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {}

  // Override this to define how to tear down the environment.
  void TearDown() override { _dfr_terminate(); }
};

extern testing::Environment *const dfr_env;
#endif
