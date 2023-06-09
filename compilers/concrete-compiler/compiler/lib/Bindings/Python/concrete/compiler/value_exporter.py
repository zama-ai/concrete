"""ValueExporter."""

# pylint: disable=no-name-in-module,import-error

from typing import List

from mlir._mlir_libs._concretelang._compiler import (
    ValueExporter as _ValueExporter,
)

from .client_parameters import ClientParameters
from .key_set import KeySet
from .value import Value
from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class ValueExporter(WrapperCpp):
    """A helper class to create `Value`s."""

    def __init__(self, value_exporter: _ValueExporter):
        """
        Wrap the native C++ object.

        Args:
            value_exporter (_ValueExporter):
                object to wrap

        Raises:
            TypeError:
                if `value_exporter` is not of type `_ValueExporter`
        """

        if not isinstance(value_exporter, _ValueExporter):
            raise TypeError(
                f"value_exporter must be of type _ValueExporter, not {type(value_exporter)}"
            )

        super().__init__(value_exporter)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(keyset: KeySet, client_parameters: ClientParameters) -> "ValueExporter":
        """
        Create a value exporter.
        """
        return ValueExporter(
            _ValueExporter.create(keyset.cpp(), client_parameters.cpp())
        )

    def export_scalar(self, position: int, value: int) -> Value:
        """
        Export scalar.

        Args:
            position (int):
                position of the argument within the circuit

            value (int):
                scalar to export

        Returns:
            Value:
                exported scalar
        """

        return Value(self.cpp().export_scalar(position, value))

    def export_tensor(
        self, position: int, values: List[int], shape: List[int]
    ) -> Value:
        """
        Export tensor.

        Args:
            position (int):
                position of the argument within the circuit

            values (List[int]):
                tensor elements to export

            shape (List[int]):
                tensor shape to export

        Returns:
            Value:
                exported tensor
        """

        return Value(self.cpp().export_tensor(position, values, shape))
