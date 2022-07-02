add_custom_target(ConcretelangClientlibTests)

add_dependencies(ConcretelangUnitTests ConcretelangClientlibTests)

add_unittest(
  ConcretelangClientlibTests

  unit_tests_concretelang_clientlib

  ClientParameters.cpp
  KeySet.cpp
)

target_link_libraries(
  unit_tests_concretelang_clientlib

  PRIVATE
  ConcretelangClientLib
  ConcretelangSupport
)
