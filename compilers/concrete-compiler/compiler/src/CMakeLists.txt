add_llvm_tool(concretecompiler main.cpp)
target_compile_options(concretecompiler PRIVATE -fexceptions)
llvm_update_compile_flags(concretecompiler)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_dependencies(concretecompiler mlir-headers)

target_link_libraries(
  concretecompiler
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          MLIRTransforms
          ConcreteDialect
          TFHEDialect
          FHEDialect
          SDFGDialect
          OptimizerDialect
          ConcretelangTransforms
          MLIRIR
          MLIRLLVMIRTransforms
          MLIRLLVMToLLVMIRTranslation
          RTDialect
          ConcretelangSupport
          ConcretelangCommon)

mlir_check_all_link_libraries(concretecompiler)
