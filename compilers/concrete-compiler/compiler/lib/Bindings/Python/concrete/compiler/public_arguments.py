#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""PublicArguments."""

from typing import List

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    PublicArguments as _PublicArguments,
)

# pylint: enable=no-name-in-module,import-error
from .client_parameters import ClientParameters
from .value import Value
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

    @staticmethod
    # pylint: disable=arguments-differ
    def new(
        client_parameters: ClientParameters, values: List[Value]
    ) -> "PublicArguments":
        """
        Create public arguments from individual values.
        """
        return PublicArguments(
            _PublicArguments.create(
                client_parameters.cpp(),
                [value.cpp() for value in values],
            )
        )

    def serialize(self) -> bytes:
        """Serialize the PublicArguments.

        Returns:
            bytes: serialized object
        """
        return self.cpp().serialize()

    @staticmethod
    def deserialize(
        client_parameters: ClientParameters, serialized_args: bytes
    ) -> "PublicArguments":
        """Unserialize PublicArguments from bytes of serialized_args.

        Args:
            client_parameters (ClientParameters): client parameters of the compiled circuit
            serialized_args (bytes): previously serialized PublicArguments

        Raises:
            TypeError: if client_parameters is not of type ClientParameters
            TypeError: if serialized_args is not of type bytes

        Returns:
            PublicArguments: deserialized object
        """
        if not isinstance(client_parameters, ClientParameters):
            raise TypeError(
                f"client_parameters must be of type ClientParameters, not {type(client_parameters)}"
            )
        if not isinstance(serialized_args, bytes):
            raise TypeError(
                f"serialized_args must be of type bytes, not {type(serialized_args)}"
            )
        return PublicArguments.wrap(
            _PublicArguments.deserialize(client_parameters.cpp(), serialized_args)
        )
