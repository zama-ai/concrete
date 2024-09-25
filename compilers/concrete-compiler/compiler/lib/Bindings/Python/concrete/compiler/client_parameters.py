#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

"""Client parameters."""

from typing import List

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    ClientParameters as _ClientParameters,
)

# pylint: enable=no-name-in-module,import-error

from .lwe_secret_key import LweSecretKeyParam
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

    def lwe_secret_key_param(self, key_id: int) -> LweSecretKeyParam:
        """Get the parameters of a selected LWE secret key.

        Args:
            key_id (int): keyid to get parameters from

        Raises:
            TypeError: if arguments aren't of expected types

        Returns:
            LweSecretKeyParam: LWE secret key parameters
        """
        if not isinstance(key_id, int):
            raise TypeError(f"key_id must be of type int, not {type(key_id)}")
        return LweSecretKeyParam.wrap(self.cpp().lwe_secret_key_param(key_id))

    def input_keyid_at(self, input_idx: int, circuit_name: str = "<lambda>") -> int:
        """Get the keyid of a selected encrypted input in a given circuit.

        Args:
            input_idx (int): index of the input in the circuit.
            circuit_name (str): name of the circuit containing the desired input.

        Raises:
            TypeError: if arguments aren't of expected types

        Returns:
            int: keyid
        """
        if not isinstance(input_idx, int):
            raise TypeError(f"input_idx must be of type int, not {type(input_idx)}")
        if not isinstance(circuit_name, str):
            raise TypeError(
                f"circuit_name must be of type str, not {type(circuit_name)}"
            )
        return self.cpp().input_keyid_at(input_idx, circuit_name)

    def input_variance_at(self, input_idx: int, circuit_name: str) -> float:
        """Get the variance of a selected encrypted input in a given circuit.

        Args:
            input_idx (int): index of the input in the circuit.
            circuit_name (str): name of the circuit containing the desired input.

        Raises:
            TypeError: if arguments aren't of expected types

        Returns:
            float: variance
        """
        if not isinstance(input_idx, int):
            raise TypeError(f"input_idx must be of type int, not {type(input_idx)}")
        if not isinstance(circuit_name, str):
            raise TypeError(
                f"circuit_name must be of type str, not {type(circuit_name)}"
            )
        return self.cpp().input_variance_at(input_idx, circuit_name)

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

    def function_list(self) -> List[str]:
        """Return the list of function names.

        Returns:
            List[str]: list of the names of the functions.
        """
        return self.cpp().function_list()

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
