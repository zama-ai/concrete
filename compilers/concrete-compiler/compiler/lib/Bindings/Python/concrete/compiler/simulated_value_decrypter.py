"""SimulatedValueDecrypter."""

# pylint: disable=no-name-in-module,import-error

from typing import List, Union

import numpy as np
from mlir._mlir_libs._concretelang._compiler import (
    SimulatedValueDecrypter as _SimulatedValueDecrypter,
)

from .client_parameters import ClientParameters
from .value import Value
from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class SimulatedValueDecrypter(WrapperCpp):
    """A helper class to decrypt `Value`s."""

    def __init__(self, value_decrypter: _SimulatedValueDecrypter):
        """
        Wrap the native C++ object.

        Args:
            value_decrypter (_SimulatedValueDecrypter):
                object to wrap

        Raises:
            TypeError:
                if `value_decrypter` is not of type `_SimulatedValueDecrypter`
        """

        if not isinstance(value_decrypter, _SimulatedValueDecrypter):
            raise TypeError(
                f"value_decrypter must be of type _SimulatedValueDecrypter, not {type(value_decrypter)}"
            )

        super().__init__(value_decrypter)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(client_parameters: ClientParameters):
        """
        Create a value decrypter.
        """
        return SimulatedValueDecrypter(
            _SimulatedValueDecrypter.create(client_parameters.cpp())
        )

    def decrypt(self, position: int, value: Value) -> Union[int, np.ndarray]:
        """
        Decrypt value.

        Args:
            position (int):
                position of the argument within the circuit

            value (Value):
                value to decrypt

        Returns:
            Union[int, np.ndarray]:
                decrypted value
        """

        shape = tuple(self.cpp().get_shape(position))

        if len(shape) == 0:
            return self.decrypt_scalar(position, value)

        return np.array(self.decrypt_tensor(position, value), dtype=np.int64).reshape(
            shape
        )

    def decrypt_scalar(self, position: int, value: Value) -> int:
        """
        Decrypt scalar.

        Args:
            position (int):
                position of the argument within the circuit

            value (Value):
                scalar value to decrypt

        Returns:
            int:
                decrypted scalar
        """

        return self.cpp().decrypt_scalar(position, value.cpp())

    def decrypt_tensor(self, position: int, value: Value) -> List[int]:
        """
        Decrypt tensor.

        Args:
            position (int):
                position of the argument within the circuit

            value (Value):
                tensor value to decrypt

        Returns:
            List[int]:
                decrypted tensor
        """

        return self.cpp().decrypt_tensor(position, value.cpp())
