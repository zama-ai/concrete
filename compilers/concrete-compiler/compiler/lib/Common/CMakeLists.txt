add_compile_options(-fexceptions -fsized-deallocation -fno-rtti)

add_library(
    _ConcreteSys_ConcretelangCommon_Sec
    STATIC
    Security.cpp
)
target_include_directories(_ConcreteSys_ConcretelangCommon_Sec PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})

add_mlir_library(
  ConcretelangCommon
  Protocol.cpp
  CRT.cpp
  Csprng.cpp
  Keys.cpp
  Keysets.cpp
  Transformers.cpp
  Security.cpp
  Values.cpp
  DEPENDS
  concrete-protocol
  LINK_LIBS
  PUBLIC
  kj
  capnp)

target_include_directories(ConcretelangCommon PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})
