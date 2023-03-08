"""
Declaration of `BaseDataType` abstract class.
"""

from abc import ABC, abstractmethod


class BaseDataType(ABC):
    """BaseDataType abstract class, to form a basis for data types."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def __str__(self) -> str:
        pass  # pragma: no cover
