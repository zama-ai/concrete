set(LLVM_OPTIONAL_SOURCES CompilerEngine.cpp)

add_mlir_public_c_api_library(ZAMALANGCAPISupport

  CompilerEngine.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  ZamalangSupport
  )
