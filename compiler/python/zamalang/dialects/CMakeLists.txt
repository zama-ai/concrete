include(AddMLIRPython)

################################################################################
# Generate dialect-specific bindings.
################################################################################

add_mlir_dialect_python_bindings(ZAMALANGBindingsPythonHLFHEOps
	TD_FILE HLFHEOps.td
	DIALECT_NAME HLFHE)
add_dependencies(ZAMALANGBindingsPythonSources ZAMALANGBindingsPythonHLFHEOps)


################################################################################
# Installation.
################################################################################

install(
  DIRECTORY ${PROJECT_BINARY_DIR}/python/zamalang/dialects
  DESTINATION python/zamalang
  COMPONENT ZAMALANGBindingsPythonDialects
  FILES_MATCHING PATTERN "_*_gen.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "__init__.py" EXCLUDE
)

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-ZAMALANGBindingsPythonDialects
    DEPENDS ZAMALANGBindingsPythonSources
    COMPONENT ZAMALANGBindingsPythonDialects)
endif()
