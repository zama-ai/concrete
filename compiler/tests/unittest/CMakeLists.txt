enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  add_compile_options(
    -DCONCRETELANG_PARALLEL_TESTING_ENABLED
    )
  link_libraries(
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Runtime
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
  )
endif()

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
  end_to_end_jit_hlfhelinalg
  end_to_end_jit_hlfhelinalg.cc
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
  end_to_end_jit_hlfhelinalg.cc
  end_to_end_jit_lambda.cc

  PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

target_link_libraries(
  end_to_end_jit_test
  gtest_main
  ConcretelangSupport
)

target_link_libraries(
  end_to_end_jit_clear_tensor
  gtest_main
  ConcretelangSupport
)

target_link_libraries(
  end_to_end_jit_encrypted_tensor
  gtest_main
  ConcretelangSupport
)

target_link_libraries(
  end_to_end_jit_hlfhelinalg
  gtest_main
  ConcretelangSupport
)

target_link_libraries(
  end_to_end_jit_lambda
  gtest_main
  ConcretelangSupport
)

include(GoogleTest)
gtest_discover_tests(end_to_end_jit_test)
gtest_discover_tests(end_to_end_jit_clear_tensor)
gtest_discover_tests(end_to_end_jit_encrypted_tensor)
gtest_discover_tests(end_to_end_jit_hlfhelinalg)
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
  target_link_libraries(
    end_to_end_jit_dfr
    gtest_main
    ConcretelangSupport
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Runtime
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
  )

  target_link_libraries(
    end_to_end_jit_auto_parallelization
    gtest_main
    ConcretelangSupport
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Runtime
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
  )

  gtest_discover_tests(end_to_end_jit_dfr)
  gtest_discover_tests(end_to_end_jit_auto_parallelization)
endif()
