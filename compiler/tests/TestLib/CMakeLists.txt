enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(
  testlib_unit_test
  testlib_unit_test.cpp
)

set_source_files_properties(
  testlib_unit_test.cpp

  PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

target_link_libraries(
  testlib_unit_test
  ConcretelangRuntime
  ConcretelangSupport
  ConcretelangClientLib
  ConcretelangServerLib
  gtest_main
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(
    testlib_unit_test
    Concrete
  )
endif()

include(GoogleTest)
gtest_discover_tests(testlib_unit_test)
