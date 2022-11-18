add_mlir_dialect_library(
  LinalgExtras
  LinalgExtras.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Concrete
  DEPENDS
  ConcreteDialect
  ConcretelangConversionPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRLinalgTransforms)

target_link_libraries(LinalgExtras PUBLIC ConcreteDialect MLIRIR)
