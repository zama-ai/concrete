"""
Declaration of `Value` class.
"""

# pylint: disable=import-error,no-name-in-module

from concrete.compiler import Value as NativeValue

# pylint: enable=import-error,no-name-in-module


class Value:
    """
    Value class, to store scalar or tensor values which can be encrypted or clear.
    """

    inner: NativeValue

    def __init__(self, inner: NativeValue):
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
    def deserialize(serialized_data: bytes) -> "Value":
        """
        Deserialize data from bytes.

        Args:
            serialized_data (bytes):
                previously serialized data

        Returns:
            Value:
                deserialized data
        """

        return Value(NativeValue.deserialize(serialized_data))
