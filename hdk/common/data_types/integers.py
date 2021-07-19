"""This file holds the definitions for integer types"""

from . import base


class Integer(base.BaseDataType):
    """Class representing an integer"""

    bit_width: int
    is_signed: bool

    def __init__(self, bit_width: int, is_signed: bool) -> None:
        self.bit_width = bit_width
        self.is_signed = is_signed

    def min_value(self) -> int:
        """Minimum value representable by the Integer"""
        if self.is_signed:
            return -(2 ** (self.bit_width - 1))

        return 0

    def max_value(self) -> int:
        """Maximum value representable by the Integer"""
        if self.is_signed:
            return 2 ** (self.bit_width - 1) - 1

        return 2 ** self.bit_width - 1

    def can_represent_value(self, value_to_represent: int) -> bool:
        """A helper function to check if a value is representable by the Integer

        Args:
            value_to_represent (int): Value to check

        Returns:
            bool: True if the value can be represented by this integer
        """
        return self.min_value() <= value_to_represent <= self.max_value()


def create_signed_integer(bit_width: int) -> Integer:
    """Convenience function to create a signed integer

    Args:
        bit_width (int): width of the integer

    Returns:
        Integer: A signed integer with the requested bit_width
    """
    return Integer(bit_width, is_signed=True)


SignedInteger = create_signed_integer


def create_unsigned_integer(bit_width: int) -> Integer:
    """Convenience function to create an unsigned integer

    Args:
        bit_width (int): width of the integer

    Returns:
        Integer: An unsigned integer with the requested bit_width
    """

    return Integer(bit_width, is_signed=False)


UnsignedInteger = create_unsigned_integer
