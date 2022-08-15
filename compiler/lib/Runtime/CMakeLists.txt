add_library(ConcretelangRuntime SHARED
  context.cpp
  wrappers.cpp
  AsyncOffload.cpp
  DFRuntime.cpp
  seeder.cpp
)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  target_link_libraries(
    ConcretelangRuntime
    PRIVATE
    HPX::hpx HPX::iostreams_component
  )
  set_source_files_properties(
    DFRuntime.cpp
    PROPERTIES COMPILE_FLAGS "-fopenmp"
  )
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(ConcretelangRuntime PUBLIC omp)
else()
  target_link_libraries(ConcretelangRuntime PUBLIC -Wl,--no-as-needed omp)
endif()

target_link_libraries(
  ConcretelangRuntime
  PUBLIC
  Concrete
  ConcretelangClientLib

  pthread m dl
  $<TARGET_OBJECTS:mlir_c_runner_utils>
)

install(TARGETS ConcretelangRuntime omp EXPORT ConcretelangRuntime)
install(EXPORT ConcretelangRuntime DESTINATION "./")
