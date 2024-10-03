"""
Declaration of `ClientSpecs` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

import json
from typing import Any

# mypy: disable-error-code=attr-defined
from concrete.compiler import ClientParameters

# pylint: enable=import-error,no-member,no-name-in-module


class ClientSpecs:
    """
    ClientSpecs class, to create Client objects.
    """

    client_parameters: ClientParameters

    def __init__(self, client_parameters: ClientParameters):
        self.client_parameters = client_parameters

    def __eq__(self, other: Any):  # pragma: no cover
        return self.client_parameters.serialize() == other.client_parameters.serialize()

    def serialize(self) -> bytes:
        """
        Serialize client specs into a string representation.

        Returns:
            bytes:
                serialized client specs
        """

        return self.client_parameters.serialize()

    @staticmethod
    def deserialize(serialized_client_specs: bytes) -> "ClientSpecs":
        """
        Create client specs from its string representation.

        Args:
            serialized_client_specs (bytes):
                client specs to deserialize

        Returns:
            ClientSpecs:
                deserialized client specs
        """

        client_parameters = ClientParameters.deserialize(serialized_client_specs)
        return ClientSpecs(client_parameters)

    def number_of_inputs(self, function_name: str) -> int:
        """
        Get number of inputs of one of the functions in the specs.
        """

        functions_parameters = json.loads(self.client_parameters.serialize())["circuits"]
        for function_parameters in functions_parameters:
            if function_parameters["name"] == function_name:
                client_parameters_json = function_parameters
                break
        else:
            message = f"Function `{function_name}` is not in the specs"
            raise ValueError(message)

        assert "inputs" in client_parameters_json
        input_specs = client_parameters_json["inputs"]

        return len(input_specs)
