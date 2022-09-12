// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/seeder.h"
#include <cassert>
#include <iostream>

#include "concrete-core-ffi.h"
#include "concretelang/Common/Error.h"

SeederBuilder *get_best_seeder() {
  SeederBuilder *builder = NULL;
  bool rdseed_seeder_available = false;
  CAPI_ASSERT_ERROR(rdseed_seeder_is_available(&rdseed_seeder_available));

  if (rdseed_seeder_available) {
    CAPI_ASSERT_ERROR(get_rdseed_seeder_builder(&builder));
    return builder;
  }

  bool unix_seeder_available = false;
  CAPI_ASSERT_ERROR(unix_seeder_is_available(&unix_seeder_available));

  if (unix_seeder_available) {
    // Security depends on /dev/random security
    uint64_t secret_high_64 = 0;
    uint64_t secret_low_64 = 0;
    CAPI_ASSERT_ERROR(
        get_unix_seeder_builder(secret_high_64, secret_low_64, &builder));

    return builder;
  }

  std::cout << "No available seeder." << std::endl;
  return builder;
}

SeederBuilder *best_seeder = get_best_seeder();
