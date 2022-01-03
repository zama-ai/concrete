add_llvm_tool(concretecompiler main.cpp)
target_compile_options(concretecompiler PRIVATE -fexceptions)
llvm_update_compile_flags(concretecompiler)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(CONCRETELANG_PARALLEL_EXECUTION_ENABLED)
  target_link_libraries(concretecompiler
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIRTransforms
    ConcreteDialect
    TFHEDialect
    FHEDialect

    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    RTDialect

    ConcretelangSupport
    ConcretelangTestLib

    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Runtime
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
    )
else()
  target_link_libraries(concretecompiler
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIRTransforms
    ConcreteDialect
    TFHEDialect
    FHEDialect

    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation

    ConcretelangSupport
    )
endif()

mlir_check_all_link_libraries(concretecompiler)
