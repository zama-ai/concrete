add_library(ZamalangRuntime SHARED
    context.c
    wrappers.c
)

target_link_libraries(ZamalangRuntime Concrete pthread m dl)

install(TARGETS ZamalangRuntime EXPORT ZamalangRuntime)
install(EXPORT ZamalangRuntime DESTINATION "./")

if(ZAMALANG_PARALLEL_EXECUTION_ENABLED)
  add_library(DFRuntime SHARED DFRuntime.cpp)
  target_link_libraries(DFRuntime PUBLIC pthread m dl HPX::hpx HPX::iostreams_component -rdynamic)

  install(TARGETS DFRuntime EXPORT DFRuntime)
  install(EXPORT DFRuntime DESTINATION "./")
endif()
