add_mlir_dialect_library(
  ConcreteToBConcrete
  ConcreteToBConcrete.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Concrete
  DEPENDS
  ConcreteDialect
  BConcreteDialect
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRMathDialect)

target_link_libraries(ConcreteToBConcrete PUBLIC BConcreteDialect MLIRIR)
