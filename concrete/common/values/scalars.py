"""Module that defines the scalar values in a program."""

from typing import Tuple

from ..data_types.base import BaseDataType
from .base import BaseValue


class ScalarValue(BaseValue):
    """Class representing a scalar value."""

    def __eq__(self, other: object) -> bool:
        return BaseValue.__eq__(self, other)

    def __str__(self) -> str:  # pragma: no cover
        encrypted_str = "Encrypted" if self._is_encrypted else "Clear"
        return f"{encrypted_str}Scalar<{self.data_type!r}>"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the ScalarValue shape property.

        Returns:
            Tuple[int, ...]: The ScalarValue shape which is `()`.
        """
        return ()


def make_clear_scalar(data_type: BaseDataType) -> ScalarValue:
    """Create a clear ScalarValue.

    Args:
        data_type (BaseDataType): The data type for the value.

    Returns:
        ScalarValue: The corresponding ScalarValue.
    """
    return ScalarValue(data_type=data_type, is_encrypted=False)


def make_encrypted_scalar(data_type: BaseDataType) -> ScalarValue:
    """Create an encrypted ScalarValue.

    Args:
        data_type (BaseDataType): The data type for the value.

    Returns:
        ScalarValue: The corresponding ScalarValue.
    """
    return ScalarValue(data_type=data_type, is_encrypted=True)


ClearScalar = make_clear_scalar
EncryptedScalar = make_encrypted_scalar
