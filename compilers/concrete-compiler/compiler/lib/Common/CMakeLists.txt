add_mlir_library(
  ConcretelangCommon
  Protocol.cpp
  CRT.cpp
  Csprng.cpp
  Keys.cpp
  Keysets.cpp
  Transformers.cpp
  Values.cpp
  DEPENDS
  concrete-protocol 
  LINK_LIBS
  PUBLIC
  concrete-protocol
  concrete_cpu)

target_include_directories(ConcretelangCommon PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})
