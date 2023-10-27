#include <gtest/gtest.h>

#include "concretelang/Common/CRT.h"
#include "tests_tools/assert.h"
namespace {
namespace crt = concretelang::crt;
typedef std::vector<int64_t> CRTModuli;

// Define a fixture for instantiate test with client parameters
class CRTTest : public ::testing::TestWithParam<CRTModuli> {};

TEST_P(CRTTest, crt_iCrt) {
  auto moduli = GetParam();

  // Max representable value from moduli
  uint64_t maxValue = 1;
  for (auto modulus : moduli)
    maxValue *= modulus;
  maxValue = maxValue - 1;

  std::vector<uint64_t> valuesToTest{0, maxValue / 2, maxValue};
  for (auto a : valuesToTest) {
    auto remainders = crt::crt(moduli, a);
    auto b = crt::iCrt(moduli, remainders);

    ASSERT_EQ(a, b);
  }
}

std::vector<CRTModuli> generateAllParameters() {
  return {
      // This is our default moduli for the 16 bits
      {7, 8, 9, 11, 13},
  };
}

INSTANTIATE_TEST_SUITE_P(CRTSuite, CRTTest,
                         ::testing::ValuesIn(generateAllParameters()),
                         [](const testing::TestParamInfo<CRTModuli> info) {
                           auto moduli = info.param;
                           std::string desc("mod");
                           if (!moduli.empty()) {
                             for (auto b : moduli) {
                               desc = desc + "_" + std::to_string(b);
                             }
                           }
                           return desc;
                         });

} // namespace
