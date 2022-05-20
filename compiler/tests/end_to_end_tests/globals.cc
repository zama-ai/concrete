#include "globals.h"
#include "tests_tools/GtestEnvironment.h"

const mlir::concretelang::V0FHEConstraint defaultV0Constraints{10, 7};

testing::Environment *const dfr_env =
    testing::AddGlobalTestEnvironment(new DFREnvironment);
