add_mlir_dialect_library(ConcreteUnparametrize
  ConcreteUnparametrize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE

  DEPENDS
  ConcreteDialect
  ConcretelangConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
)

target_link_libraries(ConcreteUnparametrize PUBLIC MLIRIR)
