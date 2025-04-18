add_compile_options(-fexceptions -fsized-deallocation -fno-rtti)

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
  LINK_LIBS
  PUBLIC
  rust_deps_bundle
  concrete-protocol
  capnp
  kj)

target_include_directories(
  ConcretelangCommon
  PUBLIC ${CONCRETE_CPU_INCLUDE_DIR}
  PUBLIC ${CONCRETE_CPU_NOISE_MODEL_INCLUDE_DIR})
