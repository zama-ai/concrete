add_compile_options(-fsized-deallocation)

add_mlir_library(
  AnalysisUtils
  Utils.cpp
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)
