enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(
  support_unit_test
  support_unit_test.cpp
)

set_source_files_properties(
  support_unit_test.cpp

  PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

target_link_libraries(
  support_unit_test
  gtest_main
  ConcretelangClientLib
)

include(GoogleTest)
gtest_discover_tests(support_unit_test)
