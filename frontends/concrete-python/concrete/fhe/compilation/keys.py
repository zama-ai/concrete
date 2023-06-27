"""
Declaration of `Keys` class.
"""

# pylint: disable=import-error,no-name-in-module

import pathlib
from pathlib import Path
from typing import Optional, Union

from concrete.compiler import ClientSupport, EvaluationKeys, KeySet, KeySetCache

from .specs import ClientSpecs

# pylint: enable=import-error,no-name-in-module


class Keys:
    """
    Keys class, to manage generate/reuse keys.

    Includes encryption keys as well as evaluation keys.
    Be careful when serializing/saving keys!
    """

    client_specs: ClientSpecs
    cache_directory: Optional[Union[str, Path]]

    _keyset_cache: Optional[KeySetCache]
    _keyset: Optional[KeySet]

    def __init__(
        self,
        client_specs: ClientSpecs,
        cache_directory: Optional[Union[str, Path]] = None,
    ):
        self.client_specs = client_specs
        self.cache_directory = cache_directory

        self._keyset_cache = None
        self._keyset = None

        if cache_directory is not None:
            self._keyset_cache = KeySetCache.new(str(cache_directory))

    def generate(self, force: bool = False, seed: Optional[int] = None):
        """
        Generate new keys.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated/loaded

            seed (Optional[int], default = None):
                seed for randomness
        """

        # seed of 0 will result in a crypto secure randomly generated 128-bit seed
        seed_msb = 0
        seed_lsb = 0

        if seed is not None:
            seed_lsb = seed & ((2**64) - 1)
            seed_msb = (seed >> 64) & ((2**64) - 1)

        if self._keyset is None or force:
            self._keyset = ClientSupport.key_set(
                self.client_specs.client_parameters,
                self._keyset_cache,
                seed_msb,
                seed_lsb,
            )

    def save(self, location: Union[str, Path]):
        """
        Save keys to a location.

        Saved keys are not encrypted, so be careful how you store/transfer them!

        Args:
            location (Union[str, Path]):
                location to save to
        """

        if not isinstance(location, Path):
            location = pathlib.Path(location)

        if location.exists():
            message = f"Unable to save keys to {location} because it already exists"
            raise ValueError(message)

        location.write_bytes(self.serialize())

    def load(self, location: Union[str, Path]):
        """
        Load keys from a location.

        Args:
            location (Union[str, Path]):
                location to load from
        """

        if not isinstance(location, Path):
            location = pathlib.Path(location)

        if not location.exists():
            message = f"Unable to load keys from {location} because it doesn't exist"
            raise ValueError(message)

        keys = Keys.deserialize(bytes(location.read_bytes()))

        self.client_specs = keys.client_specs
        self.cache_directory = None

        # pylint: disable=protected-access
        self._keyset_cache = None
        self._keyset = keys._keyset
        # pylint: enable=protected-access

    def load_if_exists_generate_and_save_otherwise(
        self,
        location: Union[str, Path],
        seed: Optional[int] = None,
    ):
        """
        Load keys from a location if they exist, else generate new keys and save to that location.

        Args:
            location (Union[str, Path]):
                location to load from or save to

            seed (Optional[int], default = None):
                seed for randomness in case keys need to be generated
        """

        if not isinstance(location, Path):
            location = pathlib.Path(location)

        if location.exists():
            self.load(location)
        else:
            self.generate(seed=seed)
            self.save(location)

    def serialize(self) -> bytes:
        """
        Serialize keys into bytes.

        Serialized keys are not encrypted, so be careful how you store/transfer them!

        Returns:
            bytes:
                serialized keys
        """

        if self._keyset is None:
            message = "Keys cannot be serialized before they are generated"
            raise RuntimeError(message)

        serialized_keyset = self._keyset.serialize()
        return serialized_keyset

    @staticmethod
    def deserialize(serialized_keys: bytes) -> "Keys":
        """
        Deserialize keys from bytes.

        Args:
            serialized_keys (bytes):
                previously serialized keys

        Returns:
            Keys:
                deserialized keys
        """

        keyset = KeySet.deserialize(serialized_keys)
        client_specs = ClientSpecs(keyset.client_parameters())

        # pylint: disable=protected-access
        result = Keys(client_specs)
        result._keyset = keyset
        # pylint: enable=protected-access

        return result

    @property
    def evaluation(self) -> EvaluationKeys:
        """
        Get only evaluation keys.
        """

        self.generate(force=False)
        assert self._keyset is not None

        return self._keyset.get_evaluation_keys()
