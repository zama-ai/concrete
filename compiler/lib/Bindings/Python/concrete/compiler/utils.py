#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""Common utils for the compiler submodule."""
import os
import numpy as np


ACCEPTED_NUMPY_UINTS = (np.uint8, np.uint16, np.uint32, np.uint64)
ACCEPTED_INTS = (int,) + ACCEPTED_NUMPY_UINTS
ACCEPTED_TYPES = (np.ndarray,) + ACCEPTED_INTS


def lookup_runtime_lib() -> str:
    """Try to find the absolute path to the runtime library.

    Returns:
        str: absolute path to the runtime library, or empty str if unsuccessful.
    """
    # Go up to site-packages level
    cwd = os.path.abspath(__file__)
    # to compiler
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    # to concrete
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    # to site-packages
    cwd = os.path.abspath(os.path.join(cwd, os.pardir))
    package_name = "concrete_compiler"
    libs_path = os.path.join(cwd, f"{package_name}.libs")
    # Can be because it's not a properly installed package
    if not os.path.exists(libs_path):
        return ""
    runtime_library_paths = [
        filename
        for filename in os.listdir(libs_path)
        if filename.startswith("libConcretelangRuntime")
    ]
    assert len(runtime_library_paths) == 1, "should be one and only one runtime library"
    return os.path.join(libs_path, runtime_library_paths[0])
