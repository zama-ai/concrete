"""
Declaration of `Float` class.
"""

from .base import BaseDataType


class Float(BaseDataType):
    """
    Float class, to represent floating point numbers.
    """

    bit_width: int

    def __init__(self, bit_width: int):
        super().__init__()

        if bit_width not in [16, 32, 64]:
            message = (
                f"Float({repr(bit_width)}) is not supported "
                f"(bit width must be one of 16, 32 or 64)"
            )
            raise ValueError(message)

        self.bit_width = bit_width

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.bit_width == other.bit_width

    def __str__(self) -> str:
        return f"float{self.bit_width}"
