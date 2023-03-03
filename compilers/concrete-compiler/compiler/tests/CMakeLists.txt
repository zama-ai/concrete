if(CONCRETELANG_UNIT_TESTS OR CONCRETELANG_BENCHMARK)
  add_subdirectory(end_to_end_fixture)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(CONCRETELANG_UNIT_TESTS)
  add_subdirectory(unit_tests)
  add_subdirectory(end_to_end_tests)
endif()

if(CONCRETELANG_BENCHMARK)
  add_subdirectory(end_to_end_benchmarks)
endif()
