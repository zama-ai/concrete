add_mlir_dialect_library(BConcreteToBConcreteCAPI
  BConcreteToBConcreteCAPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE

  DEPENDS
  BConcreteDialect
  mlir-headers

  LINK_LIBS PUBLIC
  ConcretelangConversion
  MLIRIR
  MLIRTransforms
)

target_link_libraries(BConcreteToBConcreteCAPI PUBLIC MLIRIR)
