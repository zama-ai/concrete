"""ValueDecrypter."""

# pylint: disable=no-name-in-module,import-error

from typing import Union

import numpy as np
from mlir._mlir_libs._concretelang._compiler import (
    ValueDecrypter as _ValueDecrypter,
)

from .client_parameters import ClientParameters
from .key_set import KeySet
from .value import Value
from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class ValueDecrypter(WrapperCpp):
    """A helper class to decrypt `Value`s."""

    def __init__(self, value_decrypter: _ValueDecrypter):
        """
        Wrap the native C++ object.

        Args:
            value_decrypter (_ValueDecrypter):
                object to wrap

        Raises:
            TypeError:
                if `value_decrypter` is not of type `_ValueDecrypter`
        """

        if not isinstance(value_decrypter, _ValueDecrypter):
            raise TypeError(
                f"value_decrypter must be of type _ValueDecrypter, not {type(value_decrypter)}"
            )

        super().__init__(value_decrypter)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(keyset: KeySet, client_parameters: ClientParameters):
        """
        Create a value decrypter.
        """
        return ValueDecrypter(
            _ValueDecrypter.create(keyset.cpp(), client_parameters.cpp())
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

        lambda_arg = self.cpp().decrypt(position, value.cpp())
        is_signed = lambda_arg.is_signed()
        if lambda_arg.is_scalar():
            return (
                lambda_arg.get_signed_scalar() if is_signed else lambda_arg.get_scalar()
            )

        shape = lambda_arg.get_tensor_shape()
        return (
            np.array(lambda_arg.get_signed_tensor_data()).reshape(shape)
            if is_signed
            else np.array(lambda_arg.get_tensor_data()).reshape(shape)
        )
