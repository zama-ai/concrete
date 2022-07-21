add_mlir_dialect_library(ConcreteToGPU
  ConcreteToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/Concrete

  DEPENDS
  ConcreteDialect
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  )

target_link_libraries(ConcreteToGPU PUBLIC ConcreteDialect MLIRIR)
