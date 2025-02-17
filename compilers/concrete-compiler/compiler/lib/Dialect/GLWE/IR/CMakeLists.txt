add_mlir_dialect_library(
  GLWEDialect
  GLWEDialect.cpp
  GLWEExpr.cpp
  GLWEExprDetail.cpp
  GLWEOps.cpp
  GLWETypes.cpp
  GLWEInterfaces.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/GLWE
  DEPENDS
  mlir-headers
  MLIRGLWEOpsIncGen
  GLWEInterfacesIncGen
  GLWEEnumsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
)

target_link_libraries(GLWEDialect PUBLIC MLIRIR)
