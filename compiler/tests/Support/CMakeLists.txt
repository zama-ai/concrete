add_custom_target(ConcreteCompilerSupportTests)

function(add_concretecompiler_support_test test_name)
  add_unittest(ConcreteCompilerSupportTests ${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE ConcretelangClientLib)
  set_source_files_properties(${ARGN} PROPERTIES COMPILE_FLAGS "-fno-rtti")
endfunction()

add_concretecompiler_support_test(
  support_unit_test
  support_unit_test.cpp
)
