enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)


add_executable(
  hello_test
  hello_test.cc
)
target_link_libraries(
  hello_test
  gtest_main
  ZamalangSupport
)

include(GoogleTest)
gtest_discover_tests(hello_test)

