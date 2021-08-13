include(AddMLIRPython)
add_custom_target(ZamalangBindingsPython)

################################################################################
# Build native Python extension
################################################################################

add_mlir_python_extension(ZamalangBindingsPythonExtension _zamalang
  INSTALL_DIR
    python
  SOURCES
    ZamalangModule.cpp
    HLFHEModule.cpp
    CompilerAPIModule.cpp
  LINK_LIBS
    ZAMALANGCAPIHLFHE
    ZamalangSupport
    LowLFHEDialect
    MidLFHEDialect
    HLFHEDialect
    Concrete
)
add_dependencies(ZamalangBindingsPython ZamalangBindingsPythonExtension)

################################################################################
# Copy python source tree.
################################################################################

file(GLOB_RECURSE PY_SRC_FILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/zamalang/*.py")

add_custom_target(ZAMALANGBindingsPythonSources ALL
  DEPENDS
    ${PY_SRC_FILES}
)

add_dependencies(ZamalangBindingsPython ZAMALANGBindingsPythonSources)

foreach(PY_SRC_FILE ${PY_SRC_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/${PY_SRC_FILE}")
  get_filename_component(PY_DEST_DIR "${PY_DEST_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${PY_DEST_DIR}")
  add_custom_command(
    TARGET ZAMALANGBindingsPythonSources PRE_BUILD
    COMMENT "Copying python source ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    BYPRODUCTS "${PY_DEST_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()

# Note that we copy from the source tree just like for headers because
# it will not be polluted with py_cache runtime artifacts (from testing and
# such).
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zamalang
  DESTINATION python
  COMPONENT ZAMALANGBindingsPythonSources
  FILES_MATCHING PATTERN "*.py"
)

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-ZAMALANGBindingsPythonSources
    DEPENDS ZAMALANGBindingsPythonSources
    COMPONENT ZAMALANGBindingsPythonSources)
endif()

################################################################################
# Generated sources.
################################################################################

add_subdirectory(zamalang/dialects)
