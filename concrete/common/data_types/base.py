"""File holding code to represent data types in a program."""

from abc import ABC, abstractmethod


class BaseDataType(ABC):
    """Base class to represent a data type."""

    @abstractmethod
    def __eq__(self, o: object) -> bool:
        """No default implementation."""
