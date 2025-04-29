"""
Declaration of `tfhers.Bridge` class.
"""

from typing import Optional, Union

# pylint: disable=import-error,no-member,no-name-in-module
from concrete.compiler import LweSecretKey, TfhersExporter, TfhersFheIntDescription

from concrete import fhe
from concrete.fhe.compilation.value import Value

from .dtypes import EncryptionKeyChoice, TFHERSIntegerType

# pylint: enable=import-error,no-member,no-name-in-module


class Bridge:
    """TFHErs Bridge extend a Client with TFHErs functionalities.

    client (fhe.Client): the client instance to be attached by the Bridge
    tfhers_specs (fhe.tfhers.TFHERSClientSpecs): the TFHE-rs specs of the client
    """

    client: "fhe.Client"
    tfhers_specs: "fhe.tfhers.TFHERSClientSpecs"
    _default_function: Optional[str]

    def __init__(
        self,
        client: "fhe.Client",
    ):
        assert client.specs.tfhers_specs is not None
        self.tfhers_specs = client.specs.tfhers_specs

        functions = client.specs.program_info.function_list()
        if len(functions) == 1:
            self._default_function = functions[0]
        else:
            self._default_function = None
        self.client = client

    # The following properties are for backward compatibility with the previous implementation
    @property
    def input_types_per_func(self) -> dict[str, list[Optional[TFHERSIntegerType]]]:
        """Return the input types per function map."""
        return self.tfhers_specs.input_types_per_func

    @property
    def input_shapes_per_func(self) -> dict[str, list[Optional[tuple[int, ...]]]]:
        """Return the input shapes per function map."""
        return self.tfhers_specs.input_shapes_per_func

    @property
    def output_types_per_func(self) -> dict[str, list[Optional[TFHERSIntegerType]]]:
        """Return the output types per function map."""
        return self.tfhers_specs.output_types_per_func

    @property
    def output_shapes_per_func(self) -> dict[str, list[Optional[tuple[int, ...]]]]:
        """Return the output shapes per function map."""
        return self.tfhers_specs.output_shapes_per_func

    def _get_default_func_or_raise_error(self, calling_func: str) -> str:
        if self._default_function is not None:
            return self._default_function
        msg = (
            "Module contains more than one function, so please provide 'func_name' while "
            f"calling '{calling_func}'"
        )
        raise RuntimeError(msg)

    def _input_type(self, func_name: str, input_idx: int) -> Optional[TFHERSIntegerType]:
        """Return the type of a certain input.

        Args:
            func_name (str): name of the function the input belongs to
            input_idx (int): the input index to get the type of

        Returns:
            Optional[TFHERSIntegerType]: input type. None means a non-tfhers type
        """
        return self.tfhers_specs.input_types_per_func[func_name][input_idx]

    def _output_type(self, func_name: str, output_idx: int) -> Optional[TFHERSIntegerType]:
        """Return the type of a certain output.

        Args:
            func_name (str): name of the function the output belongs to
            output_idx (int): the output index to get the type of

        Returns:
            Optional[TFHERSIntegerType]: output type. None means a non-tfhers type
        """
        return self.tfhers_specs.output_types_per_func[func_name][output_idx]

    def _input_shape(self, func_name: str, input_idx: int) -> Optional[tuple[int, ...]]:
        """Return the shape of a certain input.

        Args:
            func_name (str): name of the function the input belongs to
            input_idx (int): the input index to get the shape of

        Returns:
            Optional[Tuple[int, ...]]: input shape. None means a non-tfhers type
        """
        return self.tfhers_specs.input_shapes_per_func[func_name][input_idx]

    def _output_shape(
        self, func_name: str, output_idx: int
    ) -> Optional[tuple[int, ...]]:  # pragma: no cover
        """Return the shape of a certain output.

        Args:
            func_name (str): name of the function the output belongs to
            output_idx (int): the output index to get the shape of

        Returns:
            Optional[Tuple[int, ...]]: output shape. None means a non-tfhers type
        """
        return self.tfhers_specs.output_shapes_per_func[func_name][output_idx]

    def _input_keyid(self, func_name: str, input_idx: int) -> int:
        return self.client.specs.program_info.input_keyid_at(input_idx, func_name)

    def _input_variance(self, func_name: str, input_idx: int) -> float:
        input_type = self._input_type(func_name, input_idx)
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

        return TfhersFheIntDescription(
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

    def import_value(self, buffer: bytes, input_idx: int, func_name: Optional[str] = None) -> Value:
        """Import a serialized TFHErs integer as a Value.

        Args:
            buffer (bytes): serialized integer
            input_idx (int): the index of the input expecting this value
            func_name (Optional[str]): name of the function the value belongs to.
                Doesn't need to be provided if there is a single function.

        Returns:
            fhe.TransportValue: imported value
        """
        if func_name is None:
            func_name = self._get_default_func_or_raise_error("import_value")

        input_type = self._input_type(func_name, input_idx)
        input_shape = self._input_shape(func_name, input_idx)
        if input_type is None or input_shape is None:  # pragma: no cover
            msg = "input at 'input_idx' is not a TFHErs value"
            raise ValueError(msg)

        fheint_desc = self._description_from_type(input_type)
        keyid = self._input_keyid(func_name, input_idx)
        variance = self._input_variance(func_name, input_idx)
        return Value(TfhersExporter.import_int(buffer, fheint_desc, keyid, variance, input_shape))

    def export_value(self, value: Value, output_idx: int, func_name: Optional[str] = None) -> bytes:
        """Export a value as a serialized TFHErs integer.

        Args:
            value (TransportValue): value to export
            output_idx (int): the index corresponding to this output
            func_name (Optional[str]): name of the function the value belongs to.
                Doesn't need to be provided if there is a single function.

        Returns:
            bytes: serialized fheuint8
        """
        if func_name is None:
            func_name = self._get_default_func_or_raise_error("export_value")

        output_type = self._output_type(func_name, output_idx)
        if output_type is None:  # pragma: no cover
            msg = "output at 'output_idx' is not a TFHErs value"
            raise ValueError(msg)

        fheint_desc = self._description_from_type(output_type)
        return TfhersExporter.export_int(
            value._inner, fheint_desc  # pylint: disable=protected-access
        )

    def serialize_input_secret_key(self, input_idx: int, func_name: Optional[str] = None) -> bytes:
        """Serialize secret key used for a specific input.

        Args:
            input_idx (int): input index corresponding to the key to serialize
            func_name (Optional[str]): name of the function the key belongs to.
                Doesn't need to be provided if there is a single function.

        Returns:
            bytes: serialized key
        """
        if func_name is None:
            func_name = self._get_default_func_or_raise_error("serialize_input_secret_key")

        keyid = self._input_keyid(func_name, input_idx)
        # pylint: disable=protected-access
        keys = self.client.keys
        assert keys is not None
        secret_key = keys._keyset.get_client_keys().get_secret_keys()[keyid]  # type: ignore
        # pylint: enable=protected-access
        return secret_key.serialize()

    def keygen_with_initial_keys(
        self,
        input_idx_to_key_buffer: dict[Union[tuple[str, int], int], bytes],
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

            input_idx_to_key_buffer (Dict[Union[Tuple[str, int], int], bytes]):
                initial keys to set before keygen. Two possible formats: the first is when you have
                a single function. Here you can just provide the position of the input as index.
                The second is when you have multiple functions. You will need to provide both the
                name of the function and the input's position as index.

        Raises:
            RuntimeError: if failed to deserialize the key
        """
        initial_keys: dict[int, LweSecretKey] = {}
        for idx in input_idx_to_key_buffer:
            if isinstance(idx, tuple):
                func_name, input_idx = idx
            elif isinstance(idx, int) and self._default_function is not None:
                input_idx = idx
                func_name = self._default_function
            else:
                msg = (
                    "Module contains more than one function, so please make sure to mention "
                    "the function name (not just the position) in input_idx_to_key_buffer. "
                    "An example index would be a tuple ('my_func', 1)."
                )
                raise RuntimeError(msg)
            key_id = self._input_keyid(func_name, input_idx)
            # no need to deserialize the same key again
            if key_id in initial_keys:  # pragma: no cover
                continue

            key_buffer = input_idx_to_key_buffer[idx]
            param = self.client.specs.program_info.get_keyset_info().secret_keys()[key_id]
            try:
                initial_keys[key_id] = LweSecretKey.deserialize(key_buffer, param)
            except Exception as e:  # pragma: no cover
                msg = (
                    f"failed deserializing key for input with index {idx}. Make sure the key"
                    " is for the right input"
                )
                raise RuntimeError(msg) from e

        self.client.keygen(
            force,
            seed,
            encryption_seed,
            initial_keys,
        )


def new_bridge(
    circuit_or_module_or_client: Union["fhe.Circuit", "fhe.Module", "fhe.Client"]
) -> Bridge:
    """Create a TFHErs bridge from a circuit or module or client.

    Args:
        client (Union["fhe.Circuit", "fhe.Module", "fhe.Client"]):
            The client|circuit|module instance to be used by the Bridge.

    Returns:
        Bridge: A new Bridge instance attached to the provided client.
    """
    if isinstance(circuit_or_module_or_client, (fhe.Circuit, fhe.Module)):
        client = circuit_or_module_or_client.client
    else:
        client = circuit_or_module_or_client
    return Bridge(client)
