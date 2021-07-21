"""File holding classes representing values used by an FHE program"""

from abc import ABC

from . import base


class BaseValue(ABC):
    """Abstract base class to represent any kind of value in a program"""

    data_type: base.BaseDataType

    def __init__(self, data_type: base.BaseDataType) -> None:
        self.data_type = data_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.data_type!r}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.data_type == other.data_type


class ClearValue(BaseValue):
    """Class representing a clear/plaintext value (constant or not)"""


class EncryptedValue(BaseValue):
    """Class representing an encrypted value (constant or not)"""
