"""File holding code to represent data types in a program."""

from abc import ABC, abstractmethod
from typing import Optional, Type


class BaseDataType(ABC):
    """Base class to represent a data type."""

    # Constructor for the data type represented (for example numpy.int32 for an int32 numpy array)
    underlying_type_constructor: Optional[Type]

    def __init__(self) -> None:
        super().__init__()
        self.underlying_type_constructor = None

    @abstractmethod
    def __eq__(self, o: object) -> bool:
        """No default implementation."""
