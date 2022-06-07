#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""LibraryLambda."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    LibraryLambda as _LibraryLambda,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class LibraryLambda(WrapperCpp):
    """LibraryLambda reference a compiled library and can be ran using LibrarySupport."""

    def __init__(self, library_lambda: _LibraryLambda):
        """Wrap the native Cpp object.

        Args:
            library_lambda (_LibraryLambda): object to wrap

        Raises:
            TypeError: if library_lambda is not of type _LibraryLambda
        """
        if not isinstance(library_lambda, _LibraryLambda):
            raise TypeError(
                f"library_lambda must be of type _LibraryLambda, not {type(library_lambda)}"
            )
        super().__init__(library_lambda)
