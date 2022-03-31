#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""PublicResult."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    PublicResult as _PublicResult,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class PublicResult(WrapperCpp):
    """PublicResult holds the result of an encrypted execution and can be decrypted using ClientSupport."""

    def __init__(self, public_result: _PublicResult):
        """Wrap the native Cpp object.

        Args:
            public_result (_PublicResult): object to wrap

        Raises:
            TypeError: if public_result is not of type _PublicResult
        """
        if not isinstance(public_result, _PublicResult):
            raise TypeError(
                f"public_result must be of type _PublicResult, not {type(public_result)}"
            )
        super().__init__(public_result)
