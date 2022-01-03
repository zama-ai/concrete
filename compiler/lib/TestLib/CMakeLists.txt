add_mlir_library(ConcretelangTestLib
  Arguments.cpp
  DynamicLambda.cpp
  DynamicModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/TestLib

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC

  ConcretelangSupport
  ConcretelangClientLib
)
