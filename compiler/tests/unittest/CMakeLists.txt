enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)


add_executable(
  end_to_end_jit_test
  end_to_end_jit_clear_tensor.cc
  end_to_end_jit_encrypted_tensor.cc
  end_to_end_jit_test.cc
)
target_link_libraries(
  end_to_end_jit_test
  gtest_main
  ZamalangSupport
)

include(GoogleTest)
gtest_discover_tests(end_to_end_jit_test)

