set(LLVM_OPTIONAL_SOURCES Tracing.cpp)

add_mlir_public_c_api_library(
  CONCRETELANGCAPITRACING
  Tracing.cpp
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRCAPIIR
  TracingDialect)
