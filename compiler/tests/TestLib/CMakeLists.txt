enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  link_libraries(
    # usefull for old gcc versions
    -Wl,--allow-multiple-definition  # static concrete-optimizer and concrete shares some code
  )
endif()

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
  gtest_main
  ConcretelangSupport
)

include(GoogleTest)
gtest_discover_tests(testlib_unit_test)
