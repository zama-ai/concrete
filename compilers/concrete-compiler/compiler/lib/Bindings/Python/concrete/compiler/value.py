"""Value."""

# pylint: disable=no-name-in-module,import-error

from mlir._mlir_libs._concretelang._compiler import (
    Value as _Value,
)

from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class Value(WrapperCpp):
    """An encrypted/clear value which can be scalar/tensor."""

    def __init__(self, value: _Value):
        """
        Wrap the native C++ object.

        Args:
            value (_Value):
                object to wrap

        Raises:
            TypeError:
                if `value` is not of type `_Value`
        """

        if not isinstance(value, _Value):
            raise TypeError(f"value must be of type _Value, not {type(value)}")

        super().__init__(value)

    def serialize(self) -> bytes:
        """
        Serialize value into bytes.

        Returns:
            bytes: serialized value
        """

        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_value: bytes) -> "Value":
        """
        Deserialize value from bytes.

        Args:
            serialized_value (bytes):
                previously serialized value

        Returns:
            Value:
                deserialized value

        Raises:
            TypeError:
                if `serialized_value` is not of type `bytes`
        """

        if not isinstance(serialized_value, bytes):
            raise TypeError(
                f"serialized_value must be of type bytes, not {type(serialized_value)}"
            )

        return Value.wrap(_Value.deserialize(serialized_value))
