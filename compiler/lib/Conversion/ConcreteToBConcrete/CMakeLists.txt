add_mlir_dialect_library(ConcreteToBConcrete
  ConcreteToBConcrete.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Concrete

  DEPENDS
  ConcreteDialect
  BConcreteDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMath)

target_link_libraries(ConcreteToBConcrete PUBLIC BConcreteDialect MLIRIR)
