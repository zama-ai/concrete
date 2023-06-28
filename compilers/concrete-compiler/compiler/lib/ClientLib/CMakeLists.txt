add_mlir_library(
  ConcretelangClientLib
  ClientLambda.cpp
  ClientParameters.cpp
  EvaluationKeys.cpp
  CRT.cpp
  EncryptedArguments.cpp
  KeySet.cpp
  KeySetCache.cpp
  PublicArguments.cpp
  Serializers.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/ClientLib
  LINK_LIBS
  PUBLIC
  concrete_cpu)

target_include_directories(ConcretelangClientLib PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})

if(OUTPUT_COMPRESSION_SUPPORT)
  find_package(OpenMP REQUIRED)
  find_package(OpenSSL REQUIRED)

  add_library(libipcl.a STATIC IMPORTED)
  set_property(TARGET libipcl.a PROPERTY IMPORTED_LOCATION ${COMPRESS_LWE_INSTALL_DIR}/lib/libipcl.a)
  add_dependencies(libipcl.a compresslwe)

  add_library(libcompresslwe.a STATIC IMPORTED)
  set_property(TARGET libcompresslwe.a PROPERTY IMPORTED_LOCATION ${COMPRESS_LWE_INSTALL_DIR}/lib/libcompresslwe.a)
  add_dependencies(libcompresslwe.a compresslwe)

  target_link_libraries(ConcretelangClientLib PUBLIC libcompresslwe.a libipcl.a OpenMP::OpenMP_CXX OpenSSL::Crypto)
endif()
