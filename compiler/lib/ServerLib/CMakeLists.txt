add_compile_options( -Werror )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  add_compile_options( -Wno-error=pessimizing-move -Wno-pessimizing-move )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    add_compile_options( -Wno-error=cast-function-type -Wno-cast-function-type)
    add_compile_options( -Werror -Wno-error=pessimizing-move -Wno-pessimizing-move )
  endif()
endif()

add_mlir_library(
  ConcretelangServerLib

  DynamicRankCall.cpp
  ServerLambda.cpp
  DynamicModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ServerLib

  DEPENDS
  mlir-headers

  LINK_LIBS
  ConcretelangRuntime
  ConcretelangClientLib
)
