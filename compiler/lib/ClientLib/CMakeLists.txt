add_mlir_library(
  ConcretelangClientLib
  ClientParameters.cpp
  KeySet.cpp
  KeySetCache.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ClientLib
)