set(LLVM_TARGET_DEFINITIONS SDFGOps.td)
mlir_tablegen(SDFGEnums.h.inc -gen-enum-decls)
mlir_tablegen(SDFGEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SDFGOps.h.inc -gen-op-decls)
mlir_tablegen(SDFGOps.cpp.inc -gen-op-defs)
mlir_tablegen(SDFGTypes.h.inc -gen-typedef-decls -typedefs-dialect=SDFG)
mlir_tablegen(SDFGTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=SDFG)
mlir_tablegen(SDFGDialect.h.inc -gen-dialect-decls -dialect=SDFG)
mlir_tablegen(SDFGDialect.cpp.inc -gen-dialect-defs -dialect=SDFG)
mlir_tablegen(SDFGAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=SDFG)
mlir_tablegen(SDFGAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=SDFG)
add_public_tablegen_target(MLIRSDFGOpsIncGen)
add_dependencies(mlir-headers MLIRSDFGOpsIncGen)

add_concretelang_doc(SDFGOps SDFGDialect concretelang/ -gen-dialect-doc -dialect=SDFG)
add_concretelang_doc(SDFGOps SDFGOps concretelang/ -gen-op-doc)
add_concretelang_doc(SDFGTypes SDFGTypes concretelang/ -gen-typedef-doc)
