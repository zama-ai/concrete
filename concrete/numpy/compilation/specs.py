"""
Declaration of `ClientSpecs` class.
"""

import json
from typing import List

from concrete.compiler import ClientParameters


class ClientSpecs:
    """
    ClientSpecs class, to create Client objects.
    """

    client_parameters: ClientParameters
    output_signs: List[bool]

    def __init__(self, client_parameters: ClientParameters, output_signs: List[bool]):
        self.client_parameters = client_parameters
        self.output_signs = output_signs

    def serialize(self) -> str:
        """
        Serialize client specs into a string representation.

        Returns:
            str:
                string representation of the client specs
        """

        client_parameters_json = json.loads(self.client_parameters.serialize())
        return json.dumps(
            {
                "client_parameters": client_parameters_json,
                "output_signs": self.output_signs,
            }
        )

    @staticmethod
    def unserialize(serialized_client_specs: str) -> "ClientSpecs":
        """
        Create client specs from its string representation.

        Args:
            serialized_client_specs (str):
                client specs to unserialize

        Returns:
            ClientSpecs:
                unserialized client specs
        """

        raw_specs = json.loads(serialized_client_specs)

        client_parameters_bytes = json.dumps(raw_specs["client_parameters"]).encode("utf-8")
        client_parameters = ClientParameters.unserialize(client_parameters_bytes)

        return ClientSpecs(client_parameters, raw_specs["output_signs"])
