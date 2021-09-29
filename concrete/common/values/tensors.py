"""Module that defines the tensor values in a program."""

from math import prod
from typing import Optional, Tuple

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
        shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(dtype, is_encrypted)
        # Managing tensors as in numpy, no shape or () is treated as a 0-D array of size 1
        self._shape = shape if shape is not None else ()
        self._ndim = len(self._shape)
        self._size = prod(self._shape) if self._shape else 1

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
        return f"{encrypted_str}Tensor<{str(self.dtype)}, shape={self.shape}>"

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


def make_clear_tensor(
    dtype: BaseDataType,
    shape: Optional[Tuple[int, ...]] = None,
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
    shape: Optional[Tuple[int, ...]] = None,
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
