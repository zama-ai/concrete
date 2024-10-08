"""
Declaration of `Value` class.
"""

# pylint: disable=import-error,no-name-in-module

from concrete.compiler import TransportValue


class Value:
    """
    A public value object that can be sent between client and server.
    """

    _inner: TransportValue

    def __init__(self, inner: TransportValue):
        self._inner = inner

    @staticmethod
    def deserialize(buffer: bytes) -> "Value":
        """
        Deserialize a Value from bytes.
        """
        return Value(TransportValue.deserialize(buffer))

    def serialize(self) -> bytes:
        """
        Serialize a Value to bytes.
        """
        return self._inner.serialize()
