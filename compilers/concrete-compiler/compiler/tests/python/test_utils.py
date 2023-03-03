import re
import importlib.util
from concrete.compiler.utils import lookup_runtime_lib


def test_runtime_lib_path():
    # runtime library path should be found in case the package is installed
    compiler_spec = importlib.util.find_spec("concrete.compiler")
    # assuming installed packages should have python and site-packages as part of the path
    if compiler_spec and re.match(r".*python.*site-packages.*", compiler_spec.origin):
        runtime_lib_path = lookup_runtime_lib()
        assert isinstance(
            runtime_lib_path, str
        ), f"runtime library path should be of type str, not {type(runtime_lib_path)}"
        assert re.match(
            r".*libConcretelangRuntime.*\.(so|dylib)$", runtime_lib_path
        ), f"wrong runtime library path: {runtime_lib_path}"
