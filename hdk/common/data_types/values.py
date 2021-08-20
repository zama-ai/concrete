"""File holding classes representing values used by an FHE program."""

from abc import ABC, abstractmethod
from copy import deepcopy
from math import prod
from typing import Optional, Tuple

from .base import BaseDataType


class BaseValue(ABC):
    """Abstract base class to represent any kind of value in a program."""

    data_type: BaseDataType
    _is_encrypted: bool

    def __init__(self, data_type: BaseDataType, is_encrypted: bool) -> None:
        self.data_type = deepcopy(data_type)
        self._is_encrypted = is_encrypted

    def __repr__(self) -> str:  # pragma: no cover
        encrypted_str = "Encrypted" if self._is_encrypted else "Clear"
        return f"{encrypted_str}{self.__class__.__name__}<{self.data_type!r}>"

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.data_type == other.data_type

    @property
    def is_encrypted(self) -> bool:
        """Whether Value is encrypted or not.

        Returns:
            bool: True if encrypted False otherwise
        """
        return self._is_encrypted

    @property
    def is_clear(self) -> bool:
        """Whether Value is clear or not.

        Returns:
            bool: True if clear False otherwise
        """
        return not self._is_encrypted


class ScalarValue(BaseValue):
    """Class representing a scalar value."""

    def __eq__(self, other: object) -> bool:
        return BaseValue.__eq__(self, other)


def make_clear_scalar(
    data_type: BaseDataType,
) -> ScalarValue:
    """Helper to create a clear ScalarValue.

    Args:
        data_type (BaseDataType): The data type for the value.

    Returns:
        ScalarValue: The corresponding ScalarValue.
    """
    return ScalarValue(data_type=data_type, is_encrypted=False)


def make_encrypted_scalar(
    data_type: BaseDataType,
) -> ScalarValue:
    """Helper to create an encrypted ScalarValue.

    Args:
        data_type (BaseDataType): The data type for the value.

    Returns:
        ScalarValue: The corresponding ScalarValue.
    """
    return ScalarValue(data_type=data_type, is_encrypted=True)


ClearValue = make_clear_scalar
EncryptedValue = make_encrypted_scalar


class TensorValue(BaseValue):
    """Class representing a tensor value."""

    _shape: Tuple[int, ...]
    _ndim: int
    _size: int

    def __init__(
        self,
        data_type: BaseDataType,
        is_encrypted: bool,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(data_type, is_encrypted)
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

    @property
    def shape(self) -> Tuple[int, ...]:
        """The TensorValue shape property.

        Returns:
            Tuple[int, ...]: The TensorValue shape.
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """The TensorValue ndim property.

        Returns:
            int: The TensorValue ndim.
        """
        return self._ndim

    @property
    def size(self) -> int:
        """The TensorValue size property.

        Returns:
            int: The TensorValue size.
        """
        return self._size


def make_clear_tensor(
    data_type: BaseDataType,
    shape: Optional[Tuple[int, ...]] = None,
) -> TensorValue:
    """Helper to create a clear TensorValue.

    Args:
        data_type (BaseDataType): The data type for the tensor.
        shape (Optional[Tuple[int, ...]], optional): The tensor shape. Defaults to None.

    Returns:
        TensorValue: The corresponding TensorValue.
    """
    return TensorValue(data_type=data_type, is_encrypted=False, shape=shape)


def make_encrypted_tensor(
    data_type: BaseDataType,
    shape: Optional[Tuple[int, ...]] = None,
) -> TensorValue:
    """Helper to create an encrypted TensorValue.

    Args:
        data_type (BaseDataType): The data type for the tensor.
        shape (Optional[Tuple[int, ...]], optional): The tensor shape. Defaults to None.

    Returns:
        TensorValue: The corresponding TensorValue.
    """
    return TensorValue(data_type=data_type, is_encrypted=True, shape=shape)


ClearTensor = make_clear_tensor
EncryptedTensor = make_encrypted_tensor
