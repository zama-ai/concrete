if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_compile_options(-DCONCRETELANG_PARALLEL_EXECUTION_ENABLED)
endif()

add_library(ConcretelangRuntime SHARED
    context.cpp
    wrappers.c
)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_library(DFRuntime SHARED DFRuntime.cpp)
  target_link_libraries(DFRuntime PUBLIC pthread m dl HPX::hpx HPX::iostreams_component -rdynamic)

  install(TARGETS DFRuntime EXPORT DFRuntime)
  install(EXPORT DFRuntime DESTINATION "./")

  target_link_libraries(ConcretelangRuntime Concrete pthread m dl HPX::hpx)
else()
  target_link_libraries(ConcretelangRuntime Concrete pthread m dl)
endif()

install(TARGETS ConcretelangRuntime EXPORT ConcretelangRuntime)
install(EXPORT ConcretelangRuntime DESTINATION "./")
