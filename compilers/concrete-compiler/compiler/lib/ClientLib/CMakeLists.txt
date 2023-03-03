add_mlir_library(
  ConcretelangClientLib
  ClientLambda.cpp
  ClientParameters.cpp
  EvaluationKeys.cpp
  CRT.cpp
  EncryptedArguments.cpp
  KeySet.cpp
  KeySetCache.cpp
  PublicArguments.cpp
  Serializers.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ClientLib
  LINK_LIBS
  PUBLIC
  concrete_cpu)

target_include_directories(ConcretelangClientLib PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})
