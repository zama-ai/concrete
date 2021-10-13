"""Module that defines the values in a program."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Optional

from ..data_types.base import BaseDataType


class BaseValue(ABC):
    """Abstract base class to represent any kind of value in a program."""

    dtype: BaseDataType
    _is_encrypted: bool
    underlying_constructor: Optional[Callable]

    def __init__(self, dtype: BaseDataType, is_encrypted: bool) -> None:
        self.dtype = deepcopy(dtype)
        self._is_encrypted = is_encrypted
        self.underlying_constructor = None

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.dtype == other.dtype

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
