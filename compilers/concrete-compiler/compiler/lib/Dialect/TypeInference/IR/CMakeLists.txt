add_mlir_dialect_library(
  TypeInferenceDialect
  TypeInferenceDialect.cpp
  TypeInferenceOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/TypeInference
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)

target_link_libraries(TypeInferenceDialect PUBLIC MLIRIR)
