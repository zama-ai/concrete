include(AddMLIRPython)

################################################################################
# Decalare native Python extension
################################################################################

declare_mlir_python_sources(ZamalangBindingsPythonExtension)

declare_mlir_python_extension(ZamalangBindingsPythonExtension.Core
  MODULE_NAME _zamalang
  ADD_TO_PARENT ZamalangBindingsPythonExtension
  SOURCES
    ZamalangModule.cpp
    HLFHEModule.cpp
    CompilerAPIModule.cpp
  EMBED_CAPI_LINK_LIBS
    ZAMALANGCAPIHLFHE
    ZAMALANGCAPIHLFHELINALG
    ZAMALANGCAPISupport
)

################################################################################
# Declare python sources
################################################################################

declare_mlir_python_sources(ZamalangBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    zamalang/__init__.py
    zamalang/compiler.py
    zamalang/dialects/__init__.py
    zamalang/dialects/_ods_common.py)

################################################################################
# Declare dialect-specific bindings.
################################################################################

declare_mlir_python_sources(ZamalangBindingsPythonSources.Dialects
  ADD_TO_PARENT ZamalangBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ZamalangBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ZAMALANGBindingsPythonHLFHEOps
	TD_FILE zamalang/dialects/HLFHEOps.td
  SOURCES
    zamalang/dialects/hlfhe.py
	DIALECT_NAME HLFHE)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ZamalangBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ZAMALANGBindingsPythonHLFHELinalgOps
  TD_FILE zamalang/dialects/HLFHELinalgOps.td
  SOURCES
    zamalang/dialects/hlfhelinalg.py
  DIALECT_NAME HLFHELinalg)


################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(ZamalangBindingsPythonCAPI
  INSTALL_COMPONENT ZamalangBindingsPythonModules
  INSTALL_DESTINATION python_packages/zamalang_core/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under zamalang, this would change to
  # .../zamalang/_mlir_libs
  OUTPUT_DIRECTORY "${ZAMALANG_PYTHON_PACKAGES_DIR}/zamalang_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    # TODO: This can be chopped down significantly for size.
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    ZamalangBindingsPythonSources
    ZamalangBindingsPythonExtension
)

# Bundle the MLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the zamalang/
# folder and be accessible under the unified "import zamalang..." namespace.
add_mlir_python_modules(ZamalangMLIRPythonModules
  ROOT_PREFIX "${ZAMALANG_PYTHON_PACKAGES_DIR}/zamalang_core/mlir"
  INSTALL_PREFIX "python_packages/zamalang_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    # We need the circt extensions co-located with the MLIR extensions. When
    # the namespace is unified, this moves to the below.
    ZamalangBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    ZamalangBindingsPythonCAPI
)

# Bundle the ZAMALANG python sources into our package.
add_mlir_python_modules(ZamalangPythonModules
  ROOT_PREFIX "${ZAMALANG_PYTHON_PACKAGES_DIR}/zamalang_core"
  INSTALL_PREFIX "python_packages/zamalang_core"
  DECLARED_SOURCES
    ZamalangBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    ZamalangBindingsPythonCAPI
)