"""
Declaration of `Data` class.
"""

# pylint: disable=import-error,no-name-in-module

from concrete.compiler import Value as NativeData

# pylint: enable=import-error,no-name-in-module


class Data:
    """
    Data class, to store scalar or tensor data which can be encrypted or clear.
    """

    inner: NativeData

    def __init__(self, inner: NativeData):
        self.inner = inner

    def serialize(self) -> bytes:
        """
        Serialize data into bytes.

        Returns:
            bytes:
                serialized data
        """

        return self.inner.serialize()

    @staticmethod
    def deserialize(serialized_data: bytes) -> "Data":
        """
        Deserialize data from bytes.

        Args:
            serialized_data (bytes):
                previously serialized data

        Returns:
            Data:
                deserialized data
        """

        return Data(NativeData.deserialize(serialized_data))
