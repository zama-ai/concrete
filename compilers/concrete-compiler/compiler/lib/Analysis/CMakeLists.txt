add_compile_options(-fsized-deallocation)

add_mlir_library(
  AnalysisUtils
  StaticLoops.cpp
  Utils.cpp
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  MLIRIR)
