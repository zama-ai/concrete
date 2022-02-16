add_mlir_dialect_library(ConcreteToConcreteCAPI
  ConcreteToConcreteCAPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE

  DEPENDS
  ConcreteDialect
  ConcretelangConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
)

target_link_libraries(ConcreteToConcreteCAPI PUBLIC MLIRIR)
