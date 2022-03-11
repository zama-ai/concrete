set(LLVM_OPTIONAL_SOURCES CompilerEngine.cpp)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_compile_options(-DCONCRETELANG_PARALLEL_EXECUTION_ENABLED)
endif()

add_mlir_public_c_api_library(CONCRETELANGCAPISupport

  CompilerEngine.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  ConcretelangSupport
  )
