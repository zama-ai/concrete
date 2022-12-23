"""
Declaration of `Value` class.
"""

from typing import Any, Tuple

import numpy as np

from ..dtypes import BaseDataType, Float, Integer, UnsignedInteger


class Value:
    """
    Value class, to combine data type, shape, and encryption status into a single object.
    """

    dtype: BaseDataType
    shape: Tuple[int, ...]
    is_encrypted: bool

    @staticmethod
    def of(value: Any, is_encrypted: bool = False) -> "Value":  # pylint: disable=invalid-name
        """
        Get the `Value` that can represent `value`.

        Args:
            value (Any):
                value that needs to be represented

            is_encrypted (bool, default = False):
                whether the resulting `Value` is encrypted or not

        Returns:
            Value:
                `Value` that can represent `value`

        Raises:
            ValueError:
                if `value` cannot be represented by `Value`
        """

        # pylint: disable=too-many-branches,too-many-return-statements

        if isinstance(value, (bool, np.bool_)):
            return Value(dtype=UnsignedInteger(1), shape=(), is_encrypted=is_encrypted)

        if isinstance(value, (int, np.integer)):
            return Value(
                dtype=Integer.that_can_represent(value),
                shape=(),
                is_encrypted=is_encrypted,
            )

        if isinstance(value, (float, np.float64)):
            return Value(dtype=Float(64), shape=(), is_encrypted=is_encrypted)

        if isinstance(value, np.float32):
            return Value(dtype=Float(32), shape=(), is_encrypted=is_encrypted)

        if isinstance(value, np.float16):
            return Value(dtype=Float(16), shape=(), is_encrypted=is_encrypted)

        if isinstance(value, list):
            try:
                value = np.array(value)
            except Exception:  # pylint: disable=broad-except
                # here we try our best to convert the list to np.ndarray
                # if it fails we raise the exception at the end of the function
                pass

        if isinstance(value, np.ndarray):

            if np.issubdtype(value.dtype, np.bool_):
                return Value(dtype=UnsignedInteger(1), shape=value.shape, is_encrypted=is_encrypted)

            if np.issubdtype(value.dtype, np.integer):
                return Value(
                    dtype=Integer.that_can_represent(value),
                    shape=value.shape,
                    is_encrypted=is_encrypted,
                )

            if np.issubdtype(value.dtype, np.float64):
                return Value(dtype=Float(64), shape=value.shape, is_encrypted=is_encrypted)

            if np.issubdtype(value.dtype, np.float32):
                return Value(dtype=Float(32), shape=value.shape, is_encrypted=is_encrypted)

            if np.issubdtype(value.dtype, np.float16):
                return Value(dtype=Float(16), shape=value.shape, is_encrypted=is_encrypted)

        message = f"Value cannot represent {repr(value)}"
        raise ValueError(message)

        # pylint: enable=too-many-branches,too-many-return-statements

    def __init__(self, dtype: BaseDataType, shape: Tuple[int, ...], is_encrypted: bool):
        self.dtype = dtype
        self.shape = shape
        self.is_encrypted = is_encrypted

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Value)
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.is_encrypted == other.is_encrypted
        )

    def __str__(self) -> str:
        encrypted_or_clear_str = "Encrypted" if self.is_encrypted else "Clear"
        scalar_or_tensor_str = "Scalar" if self.is_scalar else "Tensor"
        shape_str = f", shape={self.shape}" if not self.is_scalar else ""
        return f"{encrypted_or_clear_str}{scalar_or_tensor_str}<{str(self.dtype)}{shape_str}>"

    @property
    def is_clear(self) -> bool:
        """
        Get whether the value is clear or not.

        Returns:
            bool:
                True if value is not encrypted, False otherwise
        """

        return not self.is_encrypted

    @property
    def is_scalar(self) -> bool:
        """
        Get whether the value is scalar or not.

        Returns:
            bool:
                True if shape of the value is (), False otherwise
        """

        return self.shape == ()

    @property
    def ndim(self) -> int:
        """
        Get number of dimensions of the value.

        Returns:
            int:
                number of dimensions of the value
        """

        return len(self.shape)

    @property
    def size(self) -> int:
        """
        Get number of elements in the value.

        Returns:
            int:
                number of elements in the value
        """

        return int(np.prod(self.shape))
