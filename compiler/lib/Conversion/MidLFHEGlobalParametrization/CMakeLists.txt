add_mlir_dialect_library(MidLFHEGlobalParametrization
MidLFHEGlobalParametrization.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/zamalang/Dialect/MidLFHE

  DEPENDS
  MidLFHEDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
)

target_link_libraries(MidLFHEGlobalParametrization PUBLIC MLIRIR)
