add_mlir_dialect_library(
  ConcreteToCAPI
  ConcreteToCAPI.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Concrete
  DEPENDS
  ConcreteDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms)

target_link_libraries(ConcreteToCAPI PUBLIC ConcreteDialect MLIRIR)
