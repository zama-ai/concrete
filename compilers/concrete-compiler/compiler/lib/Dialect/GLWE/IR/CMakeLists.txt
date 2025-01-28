add_mlir_dialect_library(
  GLWEDialect
  GLWEDialect.cpp
  GLWEOps.cpp
  GLWEExpr.cpp
  GLWEExprDetail.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/GLWE
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR
)

target_link_libraries(GLWEDialect PUBLIC MLIRIR)
