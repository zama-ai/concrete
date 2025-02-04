add_compile_options(-fexceptions)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    add_compile_options(-Wno-error=cast-function-type -Wno-cast-function-type)
  endif()
endif()

add_mlir_library(
  ConcretelangServerLib
  ServerLib.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ServerLib
  ${PROJECT_SOURCE_DIR}/include/concretelang/Common
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  ConcretelangCommon)
