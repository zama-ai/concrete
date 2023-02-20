add_library(ConcretelangRuntime SHARED context.cpp wrappers.cpp DFRuntime.cpp StreamEmulator.cpp)

add_dependencies(ConcretelangRuntime concrete_cpu)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  target_link_libraries(ConcretelangRuntime PRIVATE HPX::hpx HPX::iostreams_component)
  set_source_files_properties(DFRuntime.cpp PROPERTIES COMPILE_FLAGS "-fopenmp")
  target_include_directories(ConcretelangRuntime PUBLIC ${HPX_INCLUDE_DIRS})
endif()

if(CONCRETELANG_CUDA_SUPPORT)
  target_link_libraries(ConcretelangRuntime PRIVATE concrete_cuda)
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

target_include_directories(ConcretelangRuntime PUBLIC ${CONCRETE_CPU_INCLUDE_DIR})
target_link_libraries(
  ConcretelangRuntime
  PUBLIC concrete_cpu
         ConcretelangClientLib
         pthread
         m
         dl
         $<TARGET_OBJECTS:mlir_c_runner_utils>
         $<TARGET_OBJECTS:mlir_float16_utils>
         $<TARGET_OBJECTS:MLIRSparseTensorRuntime>)

install(TARGETS ConcretelangRuntime omp EXPORT ConcretelangRuntime)
install(EXPORT ConcretelangRuntime DESTINATION "./")
