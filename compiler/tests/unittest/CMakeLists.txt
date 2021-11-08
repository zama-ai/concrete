enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(
  end_to_end_jit_test
  end_to_end_jit_test.cc
  )

add_executable(
  end_to_end_jit_clear_tensor
  end_to_end_jit_clear_tensor.cc
  )

add_executable(
  end_to_end_jit_encrypted_tensor
  end_to_end_jit_encrypted_tensor.cc
  )

add_executable(
  end_to_end_jit_hlfhelinalg
  end_to_end_jit_hlfhelinalg.cc
  )

set_source_files_properties(
  end_to_end_jit_test.cc
  end_to_end_jit_clear_tensor.cc
  end_to_end_jit_encrypted_tensor.cc
  end_to_end_jit_hlfhelinalg.cc

  PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

target_link_libraries(
  end_to_end_jit_test
  gtest_main
  ZamalangSupport
)

target_link_libraries(
  end_to_end_jit_clear_tensor
  gtest_main
  ZamalangSupport
)

target_link_libraries(
  end_to_end_jit_encrypted_tensor
  gtest_main
  ZamalangSupport
)

target_link_libraries(
  end_to_end_jit_hlfhelinalg
  gtest_main
  ZamalangSupport
)

include(GoogleTest)
gtest_discover_tests(end_to_end_jit_test)
gtest_discover_tests(end_to_end_jit_clear_tensor)
gtest_discover_tests(end_to_end_jit_encrypted_tensor)
gtest_discover_tests(end_to_end_jit_hlfhelinalg)

