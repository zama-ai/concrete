add_mlir_library(HLFHEDialectAnalysis
  MANP.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/HLFHE

  DEPENDS
  HLFHEDialect
  MANPPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  HLFHEDialect)

target_link_libraries(HLFHEDialectAnalysis PUBLIC MLIRIR)
