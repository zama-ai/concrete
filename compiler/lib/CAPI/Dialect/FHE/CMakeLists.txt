set(LLVM_OPTIONAL_SOURCES FHE.cpp)

add_mlir_public_c_api_library(CONCRETELANGCAPIFHE

  FHE.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  FHEDialect
  )
