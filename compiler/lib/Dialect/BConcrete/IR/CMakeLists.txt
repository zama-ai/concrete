add_mlir_dialect_library(
  BConcreteDialect
  BConcreteDialect.cpp
  BConcreteOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/BConcrete
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(BConcreteDialect PUBLIC MLIRIR)
