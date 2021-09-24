set(LLVM_OPTIONAL_SOURCES CompilerEngine.cpp)

add_mlir_library(ZAMALANGCAPISupport

CompilerEngine.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  ZamalangSupport
  )
