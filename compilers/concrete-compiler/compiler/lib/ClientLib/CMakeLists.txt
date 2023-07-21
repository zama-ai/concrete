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

if(APPLE)
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)

  if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
  endif()
endif()

find_package(OpenMP REQUIRED)

add_library(libcompresslwe.a STATIC IMPORTED)
set_property(TARGET libcompresslwe.a PROPERTY IMPORTED_LOCATION ${COMPRESS_LWE_INSTALL_DIR}/lib/libcompresslwe.a)
add_dependencies(libcompresslwe.a compresslwe)

add_library(libhcs.a STATIC IMPORTED)
set_property(TARGET libhcs.a PROPERTY IMPORTED_LOCATION ${COMPRESS_LWE_INSTALL_DIR}/lib/libhcs.a)
add_dependencies(libhcs.a compresslwe)

# Find GMP
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY NAMES gmp)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_INCLUDE_DIR GMP_LIBRARY)
mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)
if(GMP_FOUND)
  message("GMP found")
  include_directories(${GMP_INCLUDE_DIR})
  set(EXTRA_LIBS ${EXTRA_LIBS} ${GMP_LIBRARY})
else()
  message("GMP not found")
endif()

target_link_libraries(ConcretelangClientLib PUBLIC libcompresslwe.a libhcs.a OpenMP::OpenMP_CXX gmp)
