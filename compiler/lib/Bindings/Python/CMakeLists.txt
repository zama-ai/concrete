include(AddMLIRPython)

################################################################################
# Decalare native Python extension
################################################################################

declare_mlir_python_sources(ConcretelangBindingsPythonExtension)

declare_mlir_python_extension(ConcretelangBindingsPythonExtension.Core
  MODULE_NAME _concretelang
  ADD_TO_PARENT ConcretelangBindingsPythonExtension
  SOURCES
    ConcretelangModule.cpp
    FHEModule.cpp
    CompilerAPIModule.cpp
  EMBED_CAPI_LINK_LIBS
    CONCRETELANGCAPIFHE
    CONCRETELANGCAPIFHELINALG
    CONCRETELANGCAPISupport
)

################################################################################
# Declare python sources
################################################################################

declare_mlir_python_sources(ConcretelangBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    concrete/compiler/__init__.py
    concrete/compiler/client_parameters.py
    concrete/compiler/client_support.py
    concrete/compiler/compilation_options.py
    concrete/compiler/jit_compilation_result.py
    concrete/compiler/jit_support.py
    concrete/compiler/jit_lambda.py
    concrete/compiler/key_set_cache.py
    concrete/compiler/key_set.py
    concrete/compiler/lambda_argument.py
    concrete/compiler/library_compilation_result.py
    concrete/compiler/library_support.py
    concrete/compiler/library_lambda.py
    concrete/compiler/public_arguments.py
    concrete/compiler/public_result.py
    concrete/compiler/utils.py
    concrete/compiler/wrapper.py
    concrete/__init__.py
    concrete/lang/__init__.py
    concrete/lang/dialects/__init__.py
    concrete/lang/dialects/_ods_common.py)

################################################################################
# Declare dialect-specific bindings.
################################################################################

declare_mlir_python_sources(ConcretelangBindingsPythonSources.Dialects
  ADD_TO_PARENT ConcretelangBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ConcretelangBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  CONCRETELANGBindingsPythonFHEOps
	TD_FILE concrete/lang/dialects/FHEOps.td
  SOURCES
    concrete/lang/dialects/fhe.py
	DIALECT_NAME FHE)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ConcretelangBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  CONCRETELANGBindingsPythonFHELinalgOps
  TD_FILE concrete/lang/dialects/FHELinalgOps.td
  SOURCES
    concrete/lang/dialects/fhelinalg.py
  DIALECT_NAME FHELinalg)


################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(ConcretelangBindingsPythonCAPI
  INSTALL_COMPONENT ConcretelangBindingsPythonModules
  INSTALL_DESTINATION python_packages/concretelang_core/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under concretelang, this would change to
  # .../concretelang/_mlir_libs
  OUTPUT_DIRECTORY "${CONCRETELANG_PYTHON_PACKAGES_DIR}/concretelang_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    # TODO: This can be chopped down significantly for size.
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    ConcretelangBindingsPythonSources
    ConcretelangBindingsPythonExtension
)

# Bundle the MLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the concretelang/
# folder and be accessible under the unified "import concretelang..." namespace.
add_mlir_python_modules(ConcretelangMLIRPythonModules
  ROOT_PREFIX "${CONCRETELANG_PYTHON_PACKAGES_DIR}/concretelang_core/mlir"
  INSTALL_PREFIX "python_packages/concretelang_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    # We need the circt extensions co-located with the MLIR extensions. When
    # the namespace is unified, this moves to the below.
    ConcretelangBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    ConcretelangBindingsPythonCAPI
)

# Bundle the CONCRETELANG python sources into our package.
add_mlir_python_modules(ConcretelangPythonModules
  ROOT_PREFIX "${CONCRETELANG_PYTHON_PACKAGES_DIR}/concretelang_core"
  INSTALL_PREFIX "python_packages/concretelang_core"
  DECLARED_SOURCES
    ConcretelangBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    ConcretelangBindingsPythonCAPI
)