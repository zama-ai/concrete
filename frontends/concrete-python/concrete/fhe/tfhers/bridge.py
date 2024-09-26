"""
Declaration of `tfhers.Bridge` class.
"""

from typing import Dict, List, Optional, Union

from concrete.compiler import LweSecretKey, TfhersExporter, TfhersFheIntDescription

from concrete import fhe

from .dtypes import EncryptionKeyChoice, TFHERSIntegerType


class Bridge:
    """TFHErs Bridge extend a Circuit with TFHErs functionalities.

    input_types (List[Optional[TFHERSIntegerType]]): maps every input to a type. None means
        a non-tfhers type
    output_types (List[Optional[TFHERSIntegerType]]): maps every output to a type. None means
        a non-tfhers type
    """

    circuit: "fhe.Circuit"
    input_types: List[Optional[TFHERSIntegerType]]
    output_types: List[Optional[TFHERSIntegerType]]

    def __init__(
        self,
        circuit: "fhe.Circuit",
        input_types: List[Optional[TFHERSIntegerType]],
        output_types: List[Optional[TFHERSIntegerType]],
    ):
        self.circuit = circuit
        self.input_types = input_types
        self.output_types = output_types

    def _input_type(self, input_idx: int) -> Optional[TFHERSIntegerType]:
        """Return the type of a certain input.

        Args:
            input_idx (int): the input index to get the type of

        Returns:
            Optional[TFHERSIntegerType]: input type. None means a non-tfhers type
        """
        return self.input_types[input_idx]

    def _output_type(self, output_idx: int) -> Optional[TFHERSIntegerType]:
        """Return the type of a certain output.

        Args:
            output_idx (int): the output index to get the type of

        Returns:
            Optional[TFHERSIntegerType]: output type. None means a non-tfhers type
        """
        return self.output_types[output_idx]

    def _input_keyid(self, input_idx: int) -> int:
        return self.circuit.client.specs.client_parameters.input_keyid_at(
            input_idx, self.circuit.function_name
        )

    def _input_variance(self, input_idx: int) -> float:
        input_type = self._input_type(input_idx)
        if input_type is None:  # pragma: no cover
            msg = "input at 'input_idx' is not a TFHErs value"
            raise ValueError(msg)
        return input_type.params.encryption_variance()

    @staticmethod
    def _description_from_type(
        tfhers_int_type: TFHERSIntegerType,
    ) -> TfhersFheIntDescription:
        """Construct a TFHErs integer description based on type."""

        bit_width = tfhers_int_type.bit_width
        signed = tfhers_int_type.is_signed
        params = tfhers_int_type.params
        message_modulus = 2**tfhers_int_type.msg_width
        carry_modulus = 2**tfhers_int_type.carry_width
        lwe_size = params.polynomial_size + 1
        n_cts = bit_width // tfhers_int_type.msg_width
        ks_first = params.encryption_key_choice is EncryptionKeyChoice.BIG
        # maximum value using message bits as we don't use carry bits here
        degree = message_modulus - 1
        # this should imply running a PBS on TFHErs side
        noise_level = TfhersFheIntDescription.get_unknown_noise_level()

        return TfhersFheIntDescription.new(
            bit_width,
            signed,
            message_modulus,
            carry_modulus,
            degree,
            lwe_size,
            n_cts,
            noise_level,
            ks_first,
        )

    def import_value(self, buffer: bytes, input_idx: int) -> "fhe.Value":
        """Import a serialized TFHErs integer as a Value.

        Args:
            buffer (bytes): serialized integer
            input_idx (int): the index of the input expecting this value

        Returns:
            fhe.Value: imported value
        """
        input_type = self._input_type(input_idx)
        if input_type is None:  # pragma: no cover
            msg = "input at 'input_idx' is not a TFHErs value"
            raise ValueError(msg)

        fheint_desc = self._description_from_type(input_type)

        bit_width = input_type.bit_width
        signed = input_type.is_signed
        if bit_width == 8:
            if not signed:
                keyid = self._input_keyid(input_idx)
                variance = self._input_variance(input_idx)
                return fhe.Value(
                    TfhersExporter.import_fheuint8(buffer, fheint_desc, keyid, variance)
                )

        msg = (  # pragma: no cover
            f"importing {'signed' if signed else 'unsigned'} integers of {bit_width}bits is not"
            " yet supported"
        )
        raise NotImplementedError(msg)  # pragma: no cover

    def export_value(self, value: "fhe.Value", output_idx: int) -> bytes:
        """Export a value as a serialized TFHErs integer.

        Args:
            value (fhe.Value): value to export
            output_idx (int): the index corresponding to this output

        Returns:
            bytes: serialized fheuint8
        """
        output_type = self._output_type(output_idx)
        if output_type is None:  # pragma: no cover
            msg = "output at 'output_idx' is not a TFHErs value"
            raise ValueError(msg)

        fheint_desc = self._description_from_type(output_type)

        bit_width = output_type.bit_width
        signed = output_type.is_signed
        if bit_width == 8:
            if not signed:
                return TfhersExporter.export_fheuint8(value.inner, fheint_desc)

        msg = (  # pragma: no cover
            f"exporting value to {'signed' if signed else 'unsigned'} integers of {bit_width}bits"
            " is not yet supported"
        )
        raise NotImplementedError(msg)  # pragma: no cover

    def serialize_input_secret_key(self, input_idx: int) -> bytes:
        """Serialize secret key used for a specific input.

        Args:
            input_idx (int): input index corresponding to the key to serialize

        Returns:
            bytes: serialized key
        """
        keyid = self._input_keyid(input_idx)
        # pylint: disable=protected-access
        secret_key = self.circuit.client.keys._keyset.get_lwe_secret_key(keyid)  # type: ignore
        # pylint: enable=protected-access
        return secret_key.serialize()

    def keygen_with_initial_keys(
        self,
        input_idx_to_key_buffer: Dict[int, bytes],
        force: bool = False,
        seed: Optional[int] = None,
        encryption_seed: Optional[int] = None,
    ):
        """Generate keys using an initial set of secret keys.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated

            seed (Optional[int], default = None):
                seed for private keys randomness

            encryption_seed (Optional[int], default = None):
                seed for encryption randomness

            input_idx_to_key_buffer (Dict[int, bytes]): initial keys to set before keygen

        Raises:
            RuntimeError: if failed to deserialize the key
        """
        client_specs = self.circuit.keys.client_specs
        assert isinstance(client_specs, fhe.ClientSpecs)

        initial_keys: Dict[int, LweSecretKey] = {}
        for input_idx in input_idx_to_key_buffer:
            key_id = self._input_keyid(input_idx)
            # no need to deserialize the same key again
            if key_id in initial_keys:  # pragma: no cover
                continue

            key_buffer = input_idx_to_key_buffer[input_idx]
            param = client_specs.client_parameters.lwe_secret_key_param(key_id)
            try:
                initial_keys[key_id] = LweSecretKey.deserialize(key_buffer, param)
            except Exception as e:  # pragma: no cover
                msg = (
                    f"failed deserializing key for input with index {input_idx}. Make sure the key"
                    " is for the right input"
                )
                raise RuntimeError(msg) from e

        self.circuit.keygen(
            force=force,
            seed=seed,
            encryption_seed=encryption_seed,
            initial_keys=initial_keys,
        )


def new_bridge(circuit: "fhe.Circuit") -> Bridge:
    """Create a TFHErs bridge from a circuit.

    Args:
        circuit (Circuit): compiled circuit

    Returns:
        Bridge: TFHErs bridge
    """
    input_types = [
        (
            input_node.output.dtype
            if isinstance(input_node.output.dtype, TFHERSIntegerType)
            else None
        )
        for input_node in circuit.graph.ordered_inputs()
    ]
    output_types = [
        (
            output_node.output.dtype
            if isinstance(output_node.output.dtype, TFHERSIntegerType)
            else None
        )
        for output_node in circuit.graph.ordered_outputs()
    ]

    return Bridge(circuit, input_types, output_types)
