get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(
  MLIRLowerableDialectsToLLVM
  MLIRLowerableDialectsToLLVM.cpp
  ADDITIONAL_HEADER_DIRS
  DEPENDS
  mlir-headers
  LINK_LIBS
  PUBLIC
  ${dialect_libs}
  ${conversion_libs}
  ConcretelangConversion
  MLIRIR
  MLIRTransforms
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRMathDialect)

target_link_libraries(MLIRLowerableDialectsToLLVM PUBLIC MLIRIR)
