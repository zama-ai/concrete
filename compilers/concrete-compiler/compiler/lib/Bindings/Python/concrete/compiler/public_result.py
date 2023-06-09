#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""PublicResult."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    PublicResult as _PublicResult,
)
from .client_parameters import ClientParameters

# pylint: enable=no-name-in-module,import-error
from .value import Value
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

    def n_values(self) -> int:
        """
        Get number of values in the result.
        """
        return self.cpp().n_values()

    def get_value(self, position: int) -> Value:
        """
        Get a specific value in the result.
        """
        return Value(self.cpp().get_value(position))

    def serialize(self) -> bytes:
        """Serialize the PublicResult.

        Returns:
            bytes: serialized object
        """
        return self.cpp().serialize()

    @staticmethod
    def deserialize(
        client_parameters: ClientParameters, serialized_result: bytes
    ) -> "PublicResult":
        """Unserialize PublicResult from bytes of serialized_result.

        Args:
            client_parameters (ClientParameters): client parameters of the compiled circuit
            serialized_result (bytes): previously serialized PublicResult

        Raises:
            TypeError: if client_parameters is not of type ClientParameters
            TypeError: if serialized_result is not of type bytes

        Returns:
            PublicResult: deserialized object
        """
        if not isinstance(client_parameters, ClientParameters):
            raise TypeError(
                f"client_parameters must be of type ClientParameters, not {type(client_parameters)}"
            )
        if not isinstance(serialized_result, bytes):
            raise TypeError(
                f"serialized_result must be of type bytes, not {type(serialized_result)}"
            )
        return PublicResult.wrap(
            _PublicResult.deserialize(client_parameters.cpp(), serialized_result)
        )
