add_custom_target(ConcretelangTFHETransformsTests)

add_dependencies(ConcretelangUnitTests ConcretelangTFHETransformsTests)

add_unittest(ConcretelangTFHETransformsTests unit_tests_concretelang_tfhe_transforms Parametrization.cpp)

target_link_libraries(
  unit_tests_concretelang_tfhe_transforms
  PRIVATE TFHEDialectTransforms
          MLIRParser
          MLIRExecutionEngine
          TFHEGlobalParametrization
          ConcretelangSupport
          ConcretelangClientLib
          ConcretelangServerLib
          ConcretelangCommon)
