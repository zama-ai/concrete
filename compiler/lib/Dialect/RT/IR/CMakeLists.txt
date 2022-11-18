add_mlir_dialect_library(
  RTDialect
  RTDialect.cpp
  RTOps.cpp
  RTTypes.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/RT
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(RTDialect PUBLIC MLIRIR)
