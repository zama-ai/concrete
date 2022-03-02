enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY, ${CMAKE_BINARY_DIR}/lib/)
  set (CMAKE_SHARED_LINKER_FLAGS)
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/lib/gtest_main_shared_src/TestMain.cpp.o"
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/lib/gtest_main_shared_src
    COMMAND ${CMAKE_C_COMPILER_AR} -x ${CMAKE_BINARY_DIR}/lib/libgtest_main.a --output ${CMAKE_BINARY_DIR}/lib/gtest_main_shared_src
    DEPENDS gtest_main
  )
  add_library(gtest_main_shared SHARED
    "${CMAKE_BINARY_DIR}/lib/gtest_main_shared_src/TestMain.cpp.o"
    )
  set_target_properties(gtest_main_shared PROPERTIES LINKER_LANGUAGE C)
  add_compile_options(
    -DCONCRETELANG_PARALLEL_TESTING_ENABLED
    )
  link_libraries(
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
    gtest_main_shared
    omp
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
  end_to_end_jit_fhe
  end_to_end_jit_fhe.cc
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
  end_to_end_jit_fhe
  gtest_main
  ConcretelangSupport
)

target_link_libraries(
  end_to_end_jit_fhelinalg
  gtest_main
  ConcretelangSupport
)

target_link_libraries(
  end_to_end_jit_lambda
  gtest_main
  ConcretelangSupport
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  
target_link_libraries(
  end_to_end_jit_test
  Concrete
)

target_link_libraries(
  end_to_end_jit_clear_tensor
  Concrete
)

target_link_libraries(
  end_to_end_jit_encrypted_tensor
  Concrete
)

target_link_libraries(
  end_to_end_jit_fhe
  Concrete
)

target_link_libraries(
  end_to_end_jit_fhelinalg
  Concrete
)

target_link_libraries(
  end_to_end_jit_lambda
  Concrete
)

endif()

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
  target_link_libraries(
    end_to_end_jit_dfr
    gtest_main
    ConcretelangSupport
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
  )

  target_link_libraries(
    end_to_end_jit_auto_parallelization
    gtest_main
    ConcretelangSupport
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
  )

  gtest_discover_tests(end_to_end_jit_dfr)
  gtest_discover_tests(end_to_end_jit_auto_parallelization)
endif()
