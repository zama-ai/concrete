#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""JITCompilationResult."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    JITCompilationResult as _JITCompilationResult,
)

# pylint: enable=no-name-in-module,import-error


from .wrapper import WrapperCpp


class JITCompilationResult(WrapperCpp):
    """JITCompilationResult holds the result of a JIT compilation.

    It can be instrumented using the JITSupport to load client parameters and execute the compiled
    code.
    """

    def __init__(self, jit_compilation_result: _JITCompilationResult):
        """Wrap the native Cpp object.

        Args:
            jit_compilation_result (_JITCompilationResult): object to wrap

        Raises:
            TypeError: if jit_compilation_result is not of type _JITCompilationResult
        """
        if not isinstance(jit_compilation_result, _JITCompilationResult):
            raise TypeError(
                f"jit_compilation_result must be of type _JITCompilationResult, not "
                f"{type(jit_compilation_result)}"
            )
        super().__init__(jit_compilation_result)
