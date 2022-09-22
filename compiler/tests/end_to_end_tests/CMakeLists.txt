add_custom_target(ConcreteCompilerUnitTests)

function(add_concretecompiler_unittest test_name)
  add_unittest(ConcreteCompilerUnitTests ${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE ConcretelangSupport EndToEndFixture ${RPATH_FLAGS})
  set_source_files_properties(${ARGN} PROPERTIES COMPILE_FLAGS "-fno-rtti")
endfunction()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/src/cpp)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  link_libraries(

    # usefull for old gcc versions
    -Wl,--allow-multiple-definition #  static concrete-optimizer and concrete shares some code
  )
endif()

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  add_compile_options(
    -DCONCRETELANG_DATAFLOW_TESTING_ENABLED
  )
endif()

add_concretecompiler_unittest(
  end_to_end_jit_test
  end_to_end_jit_test.cc
  globals.cc
)

add_concretecompiler_unittest(
  end_to_end_jit_encrypted_tensor
  end_to_end_jit_encrypted_tensor.cc
  globals.cc
)

add_concretecompiler_unittest(
  end_to_end_jit_fhe
  end_to_end_jit_fhe.cc
  globals.cc
)

add_concretecompiler_unittest(
  end_to_end_jit_fhelinalg
  end_to_end_jit_fhelinalg.cc
  globals.cc
)

add_concretecompiler_unittest(
  end_to_end_jit_lambda
  end_to_end_jit_lambda.cc
  globals.cc
)

if(CONCRETELANG_DATAFLOW_EXECUTION_ENABLED)
  add_concretecompiler_unittest(
    end_to_end_jit_auto_parallelization
    end_to_end_jit_auto_parallelization.cc
    globals.cc
  )
  add_concretecompiler_unittest(
    end_to_end_jit_distributed
    end_to_end_jit_distributed.cc
    globals.cc
  )
  add_concretecompiler_unittest(
    end_to_end_jit_aes_short
    end_to_end_jit_aes_short.cc
    globals.cc
  )
endif()

if(CONCRETELANG_CUDA_SUPPORT)
  add_concretecompiler_unittest(
    end_to_end_gpu_test
    end_to_end_gpu_test.cc
    globals.cc
  )
endif()
