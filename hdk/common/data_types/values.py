"""File holding classes representing values used by an FHE program."""

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

from . import base


class BaseValue(ABC):
    """Abstract base class to represent any kind of value in a program."""

    data_type: base.BaseDataType
    _is_encrypted: bool

    def __init__(self, data_type: base.BaseDataType, is_encrypted: bool) -> None:
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


ClearValue = partial(ScalarValue, is_encrypted=False)

EncryptedValue = partial(ScalarValue, is_encrypted=True)
