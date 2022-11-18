add_mlir_dialect_library(
  BConcreteToCAPI
  BConcreteToCAPI.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/BConcrete
  DEPENDS
  BConcreteDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms)

target_link_libraries(BConcreteToCAPI PUBLIC BConcreteDialect MLIRIR)
