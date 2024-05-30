add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsized-deallocation>)
add_compile_options(-Wno-unused-variable -Wno-unknown-pragmas)

if(CONCRETELANG_CUDA_SUPPORT)
  add_library(ConcretelangRuntime SHARED context.cpp simulation.cpp wrappers.cpp DFRuntime.cpp key_manager.cpp
                                         GPUDFG.cpp)
  target_link_libraries(ConcretelangRuntime PRIVATE hwloc)
else()
  # add_custom_command(
  #     OUTPUT keyswitch.tiled.c
  #     # COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/keyswitch.c keyswitch.tiled.c
  #     COMMAND polycc --second-level-tile -o keyswitch.tiled.c ${CMAKE_CURRENT_SOURCE_DIR}/keyswitch.c && sed -i 's|\#pragma|//\#pragma|g' keyswitch.tiled.c
  #     # COMMAND sed 's|\#include|//\#UNIQUEMARKERinclude|g' ${CMAKE_CURRENT_SOURCE_DIR}/keyswitch.c > keyswitch.prepared.c &&
  #     #   gcc -P -CC -E keyswitch.prepared.c -o keyswitch.preprocessed.c &&
  #     # 	polycc --tile -o keyswitch.pluto.c keyswitch.preprocessed.c &&
  #     # 	sed 's|//\#UNIQUEMARKERinclude|\#include|g' keyswitch.pluto.c > keyswitch.tiled.c &&
  #     # 	sed -i 's|\#pragma|//\#pragma|g' keyswitch.tiled.c
  #  DEPENDS keyswitch.c
  # )

  add_library(ConcretelangRuntime SHARED context.cpp simulation.cpp wrappers.cpp keyswitch.c bootstrap_optimized.c bootstrap.c tsc.c profiling.c DFRuntime.cpp key_manager.cpp StreamEmulator.cpp)
  target_link_libraries(ConcretelangRuntime PRIVATE papi fftw)

  set_source_files_properties(bootstrap_optimized.c PROPERTIES COMPILE_FLAGS "-mtune=native -march=native -ffast-math -O3")
endif()

add_dependencies(ConcretelangRuntime concrete_cpu concrete_cpu_noise_model concrete-protocol)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  target_link_libraries(ConcretelangRuntime PRIVATE HPX::hpx HPX::iostreams_component)
  set_source_files_properties(DFRuntime.cpp PROPERTIES COMPILE_FLAGS "-fopenmp")
  target_include_directories(ConcretelangRuntime PUBLIC ${HPX_INCLUDE_DIRS})
endif()

if(CONCRETELANG_CUDA_SUPPORT)
  target_link_libraries(ConcretelangRuntime LINK_PUBLIC concrete_cuda)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(ConcretelangRuntime PUBLIC omp)
else()
  target_link_libraries(ConcretelangRuntime PUBLIC -Wl,--no-as-needed omp)
endif()

# Required to link with Concrete
if(APPLE)
  find_library(SECURITY_FRAMEWORK Security)
  if(NOT SECURITY_FRAMEWORK)
    message(FATAL_ERROR "Security framework not found")
  endif()
  target_link_libraries(ConcretelangRuntime LINK_PUBLIC ${SECURITY_FRAMEWORK})
endif()

target_include_directories(
  ConcretelangRuntime
  PUBLIC ${CONCRETE_CPU_INCLUDE_DIR}
  PUBLIC ${CONCRETE_CPU_NOISE_MODEL_INCLUDE_DIR})
target_link_libraries(
  ConcretelangRuntime
  PUBLIC concrete_cpu
         concrete-protocol
         concrete_cpu_noise_model
         ConcretelangCommon
         pthread
         m
         dl
         $<TARGET_OBJECTS:mlir_c_runner_utils>
         $<TARGET_OBJECTS:mlir_float16_utils>
         $<TARGET_OBJECTS:MLIRSparseTensorRuntime>)

if(CONCRETELANG_CUDA_SUPPORT)
  install(TARGETS ConcretelangRuntime omp concrete_cuda EXPORT ConcretelangRuntime)
else()
  install(TARGETS ConcretelangRuntime omp EXPORT ConcretelangRuntime)
endif()
install(EXPORT ConcretelangRuntime DESTINATION "./")
