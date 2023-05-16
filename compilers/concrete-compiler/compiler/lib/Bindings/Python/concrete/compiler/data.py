"""
Data.
"""

# pylint: disable=import-error,no-name-in-module

from mlir._mlir_libs._concretelang._compiler import Data as _Data

from .wrapper import WrapperCpp

# pylint: enable=import-error,no-name-in-module


class Data(WrapperCpp):
    """
    An FHE data, which is either a clear or an encrypted, tensor or scalar.
    """

    def __init__(self, data: _Data):
        """
        Wrap the native Cpp object.

        Args:
            data (_Data):
                object to wrap

        Raises:
            TypeError:
                if `data` is not of type `_Data`
        """

        if not isinstance(data, _Data):
            raise TypeError(
                f"`data` must be of type `_Data`, not {type(data)}"
            )

        super().__init__(data)

    def serialize(self) -> bytes:
        """
        Serialize the Data.

        Returns:
            bytes:
                serialized data
        """

        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_data: bytes) -> "Data":
        """
        Deserialize Data from bytes.

        Args:
            serialized_data (bytes):
                previously serialized Data

        Raises:
            TypeError:
                if `serialized_data` is not of type `bytes`

        Returns:
            Data:
                deserialized data
        """

        if not isinstance(serialized_data, bytes):
            raise TypeError(
                f"`serialized_data` must be of type `bytes`, not {type(serialized_data)}"
            )

        return Data.wrap(_Data.deserialize(serialized_data))
