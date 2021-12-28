set(LLVM_OPTIONAL_SOURCES HLFHELinalg.cpp)

add_mlir_public_c_api_library(CONCRETELANGCAPIHLFHELINALG

  HLFHELinalg.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  HLFHELinalgDialect
  )
