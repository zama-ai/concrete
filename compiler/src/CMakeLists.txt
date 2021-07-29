add_llvm_tool(zamacompiler main.cpp)
target_compile_options(zamacompiler PRIVATE -fexceptions)
llvm_update_compile_flags(zamacompiler)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(zamacompiler
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}

  MLIRTransforms
  MidLFHEDialect
  HLFHEDialect

  MLIRIR
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation

  ZamalangSupport
)

mlir_check_all_link_libraries(zamacompiler)
