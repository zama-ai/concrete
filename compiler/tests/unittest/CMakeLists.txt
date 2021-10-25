enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(
  end_to_end_jit_test
  end_to_end_jit_test.cc
  )

set_source_files_properties(
  end_to_end_jit_test.cc PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_link_libraries(
  end_to_end_jit_test
  gtest_main
  ZamalangSupport
)

include(GoogleTest)
gtest_discover_tests(end_to_end_jit_test)

