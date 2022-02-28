add_compile_options( -Werror )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  add_compile_options( -Wno-error=pessimizing-move -Wno-pessimizing-move )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    add_compile_options( -Werror -Wno-error=pessimizing-move -Wno-pessimizing-move )
  endif()
endif()

add_mlir_library(
  ConcretelangClientLib
  ClientLambda.cpp
  ClientParameters.cpp
  EncryptedArguments.cpp
  KeySet.cpp
  KeySetCache.cpp
  PublicArguments.cpp
  Serializers.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ClientLib

  ConcretelangCommon
  ConcretelangRuntime
  ConcretelangSupportLib
  LINK_LIBS PUBLIC
  Concrete
)
