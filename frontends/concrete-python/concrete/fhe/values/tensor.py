"""
Declaration of `ClearTensor` and `EncryptedTensor` wrappers.
"""

from typing import Tuple

from ..dtypes import BaseDataType
from .value import Value


def clear_tensor_builder(dtype: BaseDataType, shape: Tuple[int, ...]) -> Value:
    """
    Build a clear tensor value.

    Args:
        dtype (BaseDataType):
            dtype of the value

        shape (Tuple[int, ...]):
            shape of the value

    Returns:
        Value:
            clear tensor value with given dtype and shape
    """

    return Value(dtype=dtype, shape=shape, is_encrypted=False)


ClearTensor = clear_tensor_builder


def encrypted_tensor_builder(dtype: BaseDataType, shape: Tuple[int, ...]) -> Value:
    """
    Build an encrypted tensor value.

    Args:
        dtype (BaseDataType):
            dtype of the value

        shape (Tuple[int, ...]):
            shape of the value

    Returns:
        Value:
            encrypted tensor value with given dtype and shape
    """

    return Value(dtype=dtype, shape=shape, is_encrypted=True)


EncryptedTensor = encrypted_tensor_builder
