add_custom_target(ConcreteCompilerLibTests)

function(add_concretecompiler_lib_test test_name)
  add_unittest(ConcreteCompilerLibTests ${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE ConcretelangSupport)
  set_source_files_properties(${ARGN} PROPERTIES COMPILE_FLAGS "-fno-rtti")
endfunction()

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  link_libraries(
    # usefull for old gcc versions
    -Wl,--allow-multiple-definition  # static concrete-optimizer and concrete shares some code
  )
endif()

add_concretecompiler_lib_test(
  testlib_unit_test
  testlib_unit_test.cpp
)
