enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_compile_options(
    -DCONCRETELANG_PARALLEL_TESTING_ENABLED
  )
endif()

link_libraries(
  gtest_main
  ConcretelangSupport
)

add_executable(
  end_to_end_jit_test
  end_to_end_jit_test.cc
  globals.cc
  )

add_executable(
  end_to_end_jit_clear_tensor
  end_to_end_jit_clear_tensor.cc
  globals.cc
  )

add_executable(
  end_to_end_jit_encrypted_tensor
  end_to_end_jit_encrypted_tensor.cc
  globals.cc
  )

add_executable(
  end_to_end_jit_fhe
  end_to_end_jit_fhe.cc
  EndToEndFixture.cpp
  globals.cc
  )

add_executable(
  end_to_end_jit_fhelinalg
  end_to_end_jit_fhelinalg.cc
  globals.cc
  )

add_executable(
  end_to_end_jit_lambda
  end_to_end_jit_lambda.cc
  globals.cc
)

set_source_files_properties(
  end_to_end_jit_test.cc
  end_to_end_jit_clear_tensor.cc
  end_to_end_jit_encrypted_tensor.cc
  end_to_end_jit_fhe.cc
  end_to_end_jit_fhelinalg.cc
  end_to_end_jit_lambda.cc
  EndToEndFixture.cpp

  PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

include(GoogleTest)
gtest_discover_tests(end_to_end_jit_test)
gtest_discover_tests(end_to_end_jit_clear_tensor)
gtest_discover_tests(end_to_end_jit_encrypted_tensor)
gtest_discover_tests(end_to_end_jit_fhe)
gtest_discover_tests(end_to_end_jit_fhelinalg)
gtest_discover_tests(end_to_end_jit_lambda)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_executable(
    end_to_end_jit_dfr
    end_to_end_jit_dfr.cc
  )
  add_executable(
    end_to_end_jit_auto_parallelization
    end_to_end_jit_auto_parallelization.cc
    globals.cc
  )

  set_source_files_properties(
    end_to_end_jit_dfr.cc
    end_to_end_jit_auto_parallelization.cc
    PROPERTIES COMPILE_FLAGS "-fno-rtti"
  )
  gtest_discover_tests(end_to_end_jit_dfr)
  gtest_discover_tests(end_to_end_jit_auto_parallelization)
endif()
