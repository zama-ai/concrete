add_mlir_dialect_library(
  ConcreteDialect
  ConcreteDialect.cpp
  ConcreteOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Concrete
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(ConcreteDialect PUBLIC MLIRIR)
