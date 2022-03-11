add_library(ConcretelangRuntime SHARED
    context.cpp
    wrappers.cpp
)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_library(DFRuntime SHARED DFRuntime.cpp)
  target_link_libraries(DFRuntime PUBLIC pthread m dl HPX::hpx HPX::iostreams_component -rdynamic -Wl,-wrap,main)

  install(TARGETS DFRuntime EXPORT DFRuntime)
  install(EXPORT DFRuntime DESTINATION "./")


  target_link_libraries(ConcretelangRuntime
    Concrete
    pthread
    m
    dl
    $<TARGET_OBJECTS:mlir_c_runner_utils>
    -Wl,--no-as-needed
    DFRuntime
    omp
  )
else()
  target_link_libraries(ConcretelangRuntime Concrete pthread m dl $<TARGET_OBJECTS:mlir_c_runner_utils>)
endif()

install(TARGETS ConcretelangRuntime omp EXPORT ConcretelangRuntime)
install(EXPORT ConcretelangRuntime DESTINATION "./")
