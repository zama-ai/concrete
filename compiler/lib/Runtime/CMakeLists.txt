add_library(ConcretelangRuntime SHARED
    context.c
    wrappers.c
)

target_link_libraries(ConcretelangRuntime Concrete pthread m dl)

install(TARGETS ConcretelangRuntime EXPORT ConcretelangRuntime)
install(EXPORT ConcretelangRuntime DESTINATION "./")

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_library(DFRuntime SHARED DFRuntime.cpp)
  target_link_libraries(DFRuntime PUBLIC pthread m dl HPX::hpx HPX::iostreams_component -rdynamic)

  install(TARGETS DFRuntime EXPORT DFRuntime)
  install(EXPORT DFRuntime DESTINATION "./")
endif()
