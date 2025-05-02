"""
Declaration of `TFHERSInteger` which wraps values as being of tfhers types.
"""

from typing import Union

import numpy as np

from .dtypes import TFHERSIntegerType


class TFHERSInteger:
    """Wrap integer values (scalar or arrays) into typed values, using tfhers types."""

    _value: Union[int, np.ndarray]
    _dtype: TFHERSIntegerType
    _shape: tuple

    def __init__(
        self,
        dtype: TFHERSIntegerType,
        value: Union[list, int, np.ndarray],
    ):
        if isinstance(value, list):
            try:
                value = np.array(value)
            except Exception as e:  # pylint: disable=broad-except
                msg = f"got error while trying to convert list value into a numpy array: {e}"
                raise ValueError(msg) from e
            if value.dtype == np.dtype("O"):  # pragma: no cover
                msg = "malformed value array"
                raise ValueError(msg)

        if isinstance(value, (int, np.integer)):
            self._shape = ()
        elif isinstance(value, np.ndarray):
            if value.max() > dtype.max():
                msg = "ndarray value has bigger elements than what the dtype can support"
                raise ValueError(msg)
            if value.min() < dtype.min():
                msg = "ndarray value has smaller elements than what the dtype can support"
                raise ValueError(msg)
            self._shape = value.shape
        else:
            msg = f"value can either be an int or ndarray, not a {type(value)}"
            raise TypeError(msg)

        self._value = value
        self._dtype = dtype

    @property
    def dtype(self) -> TFHERSIntegerType:
        """Get the type of the wrapped value.

        Returns:
            TFHERSIntegerType
        """
        return self._dtype

    @property
    def shape(self) -> tuple:
        """Get the shape of the wrapped value.

        Returns:
            tuple: shape
        """
        return self._shape

    @property
    def value(self) -> Union[int, np.ndarray]:
        """Get the wrapped value.

        Returns:
            Union[int, np.ndarray]
        """
        return self._value

    def min(self):
        """
        Get the minimum value that can be represented by the current type.

        Returns:
            int:
                minimum value that can be represented by the current type
        """
        return self.dtype.min()

    def max(self):
        """
        Get the maximum value that can be represented by the current type.

        Returns:
            int:
                maximum value that can be represented by the current type
        """
        return self.dtype.max()

    def __str__(self):
        return f"TFHEInteger(dtype={self.dtype}, shape={self.shape}, value={self.value})"

    def __repr__(self):
        return self.__str__()
