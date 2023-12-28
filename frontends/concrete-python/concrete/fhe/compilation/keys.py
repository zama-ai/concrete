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

    client_specs: Optional[ClientSpecs]
    cache_directory: Optional[Union[str, Path]]

    _keyset_cache: Optional[KeySetCache]
    _keyset: Optional[KeySet]

    def __init__(
        self,
        client_specs: Optional[ClientSpecs],
        cache_directory: Optional[Union[str, Path]] = None,
    ):
        self.client_specs = client_specs
        self.cache_directory = cache_directory

        self._keyset_cache = None
        self._keyset = None

        if cache_directory is not None:
            self._keyset_cache = KeySetCache.new(str(cache_directory))

    @property
    def are_generated(self) -> bool:
        """
        Get if the keys are already generated.
        """

        return self._keyset is not None

    def generate(
        self,
        force: bool = False,
        seed: Optional[int] = None,
        encryption_seed: Optional[int] = None,
    ):
        """
        Generate new keys.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated/loaded

            seed (Optional[int], default = None):
                seed for private keys randomness

            encryption_seed (Optional[int], default = None):
                seed for encryption randomness
        """

        if self._keyset is None or force:
            if self.client_specs is None:  # pragma: no cover
                message = "Tried to generate Keys without client specs."
                raise ValueError(message)
            self._keyset = ClientSupport.key_set(
                self.client_specs.client_parameters,
                self._keyset_cache,
                seed,
                encryption_seed,
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

        self.client_specs = None
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

        # pylint: disable=protected-access
        result = Keys(None)
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
