add_llvm_tool(zamacompiler main.cpp)
target_compile_options(zamacompiler PRIVATE -fexceptions)
llvm_update_compile_flags(zamacompiler)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


if(ZAMALANG_PARALLEL_EXECUTION_ENABLED)
  target_link_libraries(zamacompiler
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIRTransforms
    LowLFHEDialect
    MidLFHEDialect
    HLFHEDialect

    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation

    ZamalangSupport

    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Runtime
    -Wl,-rpath,${HPX_DIR}/../../
    -Wl,--no-as-needed
    DFRuntime
    )
else()
  target_link_libraries(zamacompiler
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIRTransforms
    LowLFHEDialect
    MidLFHEDialect
    HLFHEDialect

    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation

    ZamalangSupport
    )
endif()


mlir_check_all_link_libraries(zamacompiler)
