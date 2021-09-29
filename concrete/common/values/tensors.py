"""Module that defines the tensor values in a program."""

from math import prod
from typing import Tuple

from ..data_types.base import BaseDataType
from .base import BaseValue


class TensorValue(BaseValue):
    """Class representing a tensor value."""

    _shape: Tuple[int, ...]
    _ndim: int
    _size: int

    def __init__(
        self,
        dtype: BaseDataType,
        is_encrypted: bool,
        shape: Tuple[int, ...],
    ):
        super().__init__(dtype, is_encrypted)
        # Managing tensors as in numpy, shape of () means the value is scalar
        self._shape = shape
        self._ndim = len(self._shape)
        self._size = prod(self._shape) if self._shape != () else 1

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.shape == other.shape
            and self.ndim == other.ndim
            and self.size == other.size
            and super().__eq__(other)
        )

    def __str__(self) -> str:
        encrypted_str = "Encrypted" if self._is_encrypted else "Clear"
        tensor_or_scalar_str = "Scalar" if self.is_scalar else "Tensor"
        shape_str = f", shape={self.shape}" if self.shape != () else ""
        return f"{encrypted_str}{tensor_or_scalar_str}<{str(self.dtype)}{shape_str}>"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the TensorValue shape property.

        Returns:
            Tuple[int, ...]: The TensorValue shape.
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """Return the TensorValue ndim property.

        Returns:
            int: The TensorValue ndim.
        """
        return self._ndim

    @property
    def size(self) -> int:
        """Return the TensorValue size property.

        Returns:
            int: The TensorValue size.
        """
        return self._size

    @property
    def is_scalar(self) -> bool:
        """Whether Value is scalar or not.

        Returns:
            bool: True if scalar False otherwise
        """
        return self.shape == ()


def make_clear_tensor(
    dtype: BaseDataType,
    shape: Tuple[int, ...],
) -> TensorValue:
    """Create a clear TensorValue.

    Args:
        dtype (BaseDataType): The data type for the tensor.
        shape (Optional[Tuple[int, ...]], optional): The tensor shape. Defaults to None.

    Returns:
        TensorValue: The corresponding TensorValue.
    """
    return TensorValue(dtype=dtype, is_encrypted=False, shape=shape)


def make_encrypted_tensor(
    dtype: BaseDataType,
    shape: Tuple[int, ...],
) -> TensorValue:
    """Create an encrypted TensorValue.

    Args:
        dtype (BaseDataType): The data type for the tensor.
        shape (Optional[Tuple[int, ...]], optional): The tensor shape. Defaults to None.

    Returns:
        TensorValue: The corresponding TensorValue.
    """
    return TensorValue(dtype=dtype, is_encrypted=True, shape=shape)


ClearTensor = make_clear_tensor
EncryptedTensor = make_encrypted_tensor


def make_clear_scalar(dtype: BaseDataType) -> TensorValue:
    """Create a clear scalar value.

    Args:
        dtype (BaseDataType): The data type for the value.

    Returns:
        TensorValue: The corresponding TensorValue.
    """
    return TensorValue(dtype=dtype, is_encrypted=False, shape=())


def make_encrypted_scalar(dtype: BaseDataType) -> TensorValue:
    """Create an encrypted scalar value.

    Args:
        dtype (BaseDataType): The data type for the value.

    Returns:
        TensorValue: The corresponding TensorValue.
    """
    return TensorValue(dtype=dtype, is_encrypted=True, shape=())


ClearScalar = make_clear_scalar
EncryptedScalar = make_encrypted_scalar
