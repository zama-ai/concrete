#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt for license information.

"""PublicArguments."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    PublicArguments as _PublicArguments,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class PublicArguments(WrapperCpp):
    """PublicArguments holds encrypted and plain arguments, as well as public materials.

    An encrypted computation may require both encrypted and plain arguments, PublicArguments holds both
    types, but also other public materials, such as public keys, which are required for private computation.
    """

    def __init__(self, public_arguments: _PublicArguments):
        """Wrap the native Cpp object.

        Args:
            public_arguments (_PublicArguments): object to wrap

        Raises:
            TypeError: if public_arguments is not of type _PublicArguments
        """
        if not isinstance(public_arguments, _PublicArguments):
            raise TypeError(
                f"public_arguments must be of type _PublicArguments, not {type(public_arguments)}"
            )
        super().__init__(public_arguments)
