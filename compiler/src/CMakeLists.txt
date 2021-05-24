add_llvm_tool(zamacompiler main.cpp)
target_compile_options(zamacompiler PRIVATE -fexceptions)
llvm_update_compile_flags(zamacompiler)
target_link_libraries(zamacompiler
  PRIVATE
  MLIRTransforms
  HLFHEDialect)

mlir_check_all_link_libraries(zamacompiler)
