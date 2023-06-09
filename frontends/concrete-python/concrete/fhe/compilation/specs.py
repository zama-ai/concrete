"""
Declaration of `ClientSpecs` class.
"""

# pylint: disable=import-error,no-member,no-name-in-module

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

    def __eq__(self, other: Any):
        if self.client_parameters.serialize() != other.client_parameters.serialize():
            return False

        return True

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
