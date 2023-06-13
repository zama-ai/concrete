"""SimulatedValueExporter."""

# pylint: disable=no-name-in-module,import-error

from typing import List

from mlir._mlir_libs._concretelang._compiler import (
    SimulatedValueExporter as _SimulatedValueExporter,
)

from .client_parameters import ClientParameters
from .value import Value
from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class SimulatedValueExporter(WrapperCpp):
    """A helper class to create `Value`s."""

    def __init__(self, value_exporter: _SimulatedValueExporter):
        """
        Wrap the native C++ object.

        Args:
            value_exporter (_SimulatedValueExporter):
                object to wrap

        Raises:
            TypeError:
                if `value_exporter` is not of type `_SimulatedValueExporter`
        """

        if not isinstance(value_exporter, _SimulatedValueExporter):
            raise TypeError(
                f"value_exporter must be of type _SimulatedValueExporter, not {type(value_exporter)}"
            )

        super().__init__(value_exporter)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(client_parameters: ClientParameters) -> "SimulatedValueExporter":
        """
        Create a value exporter.
        """
        return SimulatedValueExporter(
            _SimulatedValueExporter.create(client_parameters.cpp())
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
