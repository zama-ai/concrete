#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""JITLambda."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    JITLambda as _JITLambda,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class JITLambda(WrapperCpp):
    """JITLambda contains an in-memory executable code and can be ran using JITSupport.

    It's an artifact of JIT compilation, which stays in memory and can be executed with the help of
    JITSupport.
    """

    def __init__(self, jit_lambda: _JITLambda):
        """Wrap the native Cpp object.

        Args:
            jit_lambda (_JITLambda): object to wrap

        Raises:
            TypeError: if jit_lambda is not of type JITLambda
        """
        if not isinstance(jit_lambda, _JITLambda):
            raise TypeError(
                f"jit_lambda must be of type _JITLambda, not {type(jit_lambda)}"
            )
        super().__init__(jit_lambda)
