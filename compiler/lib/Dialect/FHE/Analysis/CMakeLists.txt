add_mlir_library(FHEDialectAnalysis
  MANP.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/FHE

  DEPENDS
  FHEDialect
  MANPPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  FHEDialect)

target_link_libraries(FHEDialectAnalysis PUBLIC MLIRIR)
