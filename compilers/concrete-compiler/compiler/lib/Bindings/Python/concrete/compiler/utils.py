#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""Common utils for the compiler submodule."""
import os
import numpy as np


ACCEPTED_NUMPY_UINTS = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
ACCEPTED_INTS = (int,) + ACCEPTED_NUMPY_UINTS
ACCEPTED_TYPES = (np.ndarray,) + ACCEPTED_INTS


def lookup_runtime_lib() -> str:
    """Try to find the absolute path to the runtime library.

    Returns:
        str: absolute path to the runtime library, or empty str if unsuccessful.
    """
    # Linux and MacOS store the runtime lib at two different locations
    lib_dir = _lookup_runtime_lib_dir_linux()
    if not os.path.exists(lib_dir):
        lib_dir = _lookup_runtime_lib_dir_macos()

    # Can be because it's not a properly installed package
    if not os.path.exists(lib_dir):
        return ""
    runtime_library_paths = [
        filename
        for filename in os.listdir(lib_dir)
        if filename.startswith("libConcretelangRuntime")
    ]
    assert len(runtime_library_paths) == 1, "should be one and only one runtime library"
    return os.path.join(lib_dir, runtime_library_paths[0])


def _lookup_runtime_lib_dir_linux() -> str:
    # Go up to site-packages level
    cwd = os.path.abspath(__file__)
    # to compiler
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    # to concrete
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    # to site-packages
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))

    possible_package_names = ["concrete_python", "concrete_compiler"]
    for name in possible_package_names:
        candidate = os.path.join(cwd, f"{name}.libs")
        if os.path.exists(candidate):
            return candidate

    return ""


def _lookup_runtime_lib_dir_macos() -> str:
    # Go up to the concrete package level
    cwd = os.path.abspath(__file__)
    # to compiler
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    # to concrete
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    return os.path.join(cwd, ".dylibs")
