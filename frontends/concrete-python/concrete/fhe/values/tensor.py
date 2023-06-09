"""
Declaration of `ClearTensor` and `EncryptedTensor` wrappers.
"""

from typing import Tuple

from ..dtypes import BaseDataType
from .value_description import ValueDescription


def clear_tensor_builder(dtype: BaseDataType, shape: Tuple[int, ...]) -> ValueDescription:
    """
    Build a clear tensor value.

    Args:
        dtype (BaseDataType):
            dtype of the value

        shape (Tuple[int, ...]):
            shape of the value

    Returns:
        ValueDescription:
            clear tensor value description with given dtype and shape
    """

    return ValueDescription(dtype=dtype, shape=shape, is_encrypted=False)


ClearTensor = clear_tensor_builder


def encrypted_tensor_builder(dtype: BaseDataType, shape: Tuple[int, ...]) -> ValueDescription:
    """
    Build an encrypted tensor value.

    Args:
        dtype (BaseDataType):
            dtype of the value

        shape (Tuple[int, ...]):
            shape of the value

    Returns:
        ValueDescription:
            encrypted tensor value description with given dtype and shape
    """

    return ValueDescription(dtype=dtype, shape=shape, is_encrypted=True)


EncryptedTensor = encrypted_tensor_builder
