#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""Client parameters."""

from typing import List

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    ClientParameters as _ClientParameters,
)

# pylint: enable=no-name-in-module,import-error

from .wrapper import WrapperCpp


class ClientParameters(WrapperCpp):
    """ClientParameters are public parameters used for key generation.

    It's a compilation artifact that describes which and how public and private keys should be generated,
    and used to encrypt arguments of the compiled function.
    """

    def __init__(self, client_parameters: _ClientParameters):
        """Wrap the native Cpp object.

        Args:
            client_parameters (_ClientParameters): object to wrap

        Raises:
            TypeError: if client_parameters is not of type _ClientParameters
        """
        if not isinstance(client_parameters, _ClientParameters):
            raise TypeError(
                f"client_parameters must be of type _ClientParameters, not {type(client_parameters)}"
            )
        super().__init__(client_parameters)

    def input_signs(self) -> List[bool]:
        """Return the sign information of inputs.

        Returns:
            List[bool]: list of booleans to indicate whether the inputs are signed or not
        """
        return self.cpp().input_signs()

    def output_signs(self) -> List[bool]:
        """Return the sign information of outputs.

        Returns:
            List[bool]: list of booleans to indicate whether the outputs are signed or not
        """
        return self.cpp().output_signs()

    def serialize(self) -> bytes:
        """Serialize the ClientParameters.

        Returns:
            bytes: serialized object
        """
        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_params: bytes) -> "ClientParameters":
        """Unserialize ClientParameters from bytes of serialized_params.

        Args:
            serialized_params (bytes): previously serialized ClientParameters

        Raises:
            TypeError: if serialized_params is not of type bytes

        Returns:
            ClientParameters: deserialized object
        """
        if not isinstance(serialized_params, bytes):
            raise TypeError(
                f"serialized_params must be of type bytes, not {type(serialized_params)}"
            )
        return ClientParameters.wrap(_ClientParameters.deserialize(serialized_params))
