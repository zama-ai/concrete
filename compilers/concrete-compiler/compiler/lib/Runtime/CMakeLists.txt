add_compile_options(-fsized-deallocation)

if(CONCRETELANG_CUDA_SUPPORT)
  add_library(
    ConcretelangRuntime SHARED
    context.cpp
    DFRuntime.cpp
    key_manager.cpp
    GPUDFG.cpp
    simulation.cpp
    wrappers.cpp
    time_util.cpp)
else()
  add_library(
    ConcretelangRuntime SHARED
    context.cpp
    DFRuntime.cpp
    key_manager.cpp
    GPUDFG.cpp
    simulation.cpp
    wrappers.cpp
    time_util.cpp)
endif()

add_dependencies(ConcretelangRuntime rust_deps_bundle concrete-protocol)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  target_link_libraries(ConcretelangRuntime PUBLIC HPX::hpx HPX::iostreams_component HPX::component)
  target_link_libraries(ConcretelangRuntime PRIVATE hwloc)
  set_source_files_properties(DFRuntime.cpp PROPERTIES COMPILE_FLAGS "-fopenmp")
  add_dependencies(ConcretelangRuntime HPXLibs)
endif()

if(CONCRETELANG_CUDA_SUPPORT)
  target_link_libraries(ConcretelangRuntime LINK_PUBLIC tfhe_cuda_backend)
  target_link_libraries(ConcretelangRuntime PRIVATE hwloc)
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
  PRIVATE pthread
          m
          dl
          $<TARGET_OBJECTS:mlir_c_runner_utils>
          $<TARGET_OBJECTS:mlir_float16_utils>
          $<TARGET_OBJECTS:MLIRSparseTensorRuntime>
          $<LINK_LIBRARY:WHOLE_ARCHIVE,ConcretelangCommon>)

if(CONCRETELANG_CUDA_SUPPORT)
  install(TARGETS ConcretelangRuntime omp tfhe_cuda_backend EXPORT ConcretelangRuntime)
else()
  install(TARGETS ConcretelangRuntime omp EXPORT ConcretelangRuntime)
endif()
install(EXPORT ConcretelangRuntime DESTINATION "./")

#

add_library(
  ConcretelangRuntimeStatic STATIC
  context.cpp
  DFRuntime.cpp
  key_manager.cpp
  GPUDFG.cpp
  simulation.cpp
  wrappers.cpp
  time_util.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(ConcretelangRuntimeStatic PUBLIC omp)
else()
  target_link_libraries(ConcretelangRuntimeStatic PUBLIC -Wl,--no-as-needed omp)
endif()

target_include_directories(
  ConcretelangRuntimeStatic
  PUBLIC ${CONCRETE_CPU_INCLUDE_DIR}
  PUBLIC ${CONCRETE_CPU_NOISE_MODEL_INCLUDE_DIR})

target_link_libraries(
  ConcretelangRuntimeStatic
  PUBLIC rust_deps_bundle
         concrete-protocol
         pthread
         m
         dl
         $<TARGET_OBJECTS:mlir_c_runner_utils>
         $<TARGET_OBJECTS:mlir_float16_utils>
         $<TARGET_OBJECTS:MLIRSparseTensorRuntime>)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  target_link_libraries(ConcretelangRuntimeStatic PUBLIC HPX::hpx HPX::iostreams_component HPX::component)
endif()
