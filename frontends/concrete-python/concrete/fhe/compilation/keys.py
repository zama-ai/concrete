"""
Declaration of `Keys` class.
"""

# pylint: disable=import-error,no-name-in-module

import pathlib
from pathlib import Path
from typing import Optional, Union

from concrete.compiler import Keyset, KeysetCache, LweSecretKey

from .evaluation_keys import EvaluationKeys
from .specs import ClientSpecs

# pylint: enable=import-error,no-name-in-module


class Keys:
    """
    Keys class, to manage generate/reuse keys.

    Includes encryption keys as well as evaluation keys.
    Be careful when serializing/saving keys!
    """

    _cache: Optional[KeysetCache]
    _specs: Optional[ClientSpecs]
    _keyset: Optional[Keyset]

    def __init__(
        self,
        specs: Optional[ClientSpecs],
        cache_directory: Optional[Union[str, Path]] = None,
    ):
        self._cache = KeysetCache(str(cache_directory)) if cache_directory is not None else None
        self._specs = specs
        self._keyset = None

    @property
    def are_generated(self) -> bool:
        """
        Get if the keys are already generated.
        """

        return self._keyset is not None

    def generate(
        self,
        force: bool = False,
        secret_seed: Optional[int] = None,
        encryption_seed: Optional[int] = None,
        initial_keys: Optional[dict[int, LweSecretKey]] = None,
    ):
        """
        Generate new keys.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated/loaded

            secret_seed (Optional[int], default = None):
                seed for private keys randomness

            encryption_seed (Optional[int], default = None):
                seed for encryption randomness

            initial_keys (Optional[Dict[int, LweSecretKey]] = None):
                initial keys to set before keygen
        """

        if self._keyset is None or force:
            if self._specs is None:  # pragma: no cover
                message = "Tried to generate Keys without client specs."
                raise ValueError(message)

            secret_seed = 0 if secret_seed is None else secret_seed
            encryption_seed = 0 if encryption_seed is None else encryption_seed
            if secret_seed < 0 or secret_seed >= 2**128:
                message = "secret_seed must be a positive 128 bits integer"
                raise ValueError(message)
            if encryption_seed < 0 or encryption_seed >= 2**128:
                message = "encryption_seed must be a positive 128 bits integer"
                raise ValueError(message)
            secret_seed_msb = (secret_seed >> 64) & 0xFFFFFFFFFFFFFFFF
            secret_seed_lsb = (secret_seed) & 0xFFFFFFFFFFFFFFFF
            encryption_seed_msb = (encryption_seed >> 64) & 0xFFFFFFFFFFFFFFFF
            encryption_seed_lsb = (encryption_seed) & 0xFFFFFFFFFFFFFFFF

            self._keyset = Keyset(
                self._specs.program_info,
                self._cache,
                secret_seed_msb,
                secret_seed_lsb,
                encryption_seed_msb,
                encryption_seed_lsb,
                initial_keys,
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

        self.serialize_to_file(location)

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

        keys = Keys.deserialize(location)

        # pylint: disable=protected-access
        self._specs = None
        self._cache = None
        self._keyset = keys._keyset
        # pylint: enable=protected-access

    def load_from_bytes(self, serialized_keys: bytes):
        """
        Load keys from bytes.

        Args:
            serialized_keys (bytes): serialized keys to load from
        """

        keys = Keys.deserialize(serialized_keys)

        # pylint: disable=protected-access
        self._keyset = keys._keyset
        # pylint: enable=protected-access

    def load_if_exists_generate_and_save_otherwise(
        self,
        location: Union[str, Path],
        secret_seed: Optional[int] = None,
    ):
        """
        Load keys from a location if they exist, else generate new keys and save to that location.

        Args:
            location (Union[str, Path]):
                location to load from or save to

            secret_seed (Optional[int], default = None):
                seed for randomness in case keys need to be generated
        """

        if not isinstance(location, Path):
            location = pathlib.Path(location)

        if location.exists():
            self.load(location)
        else:
            self.generate(secret_seed=secret_seed)
            self.save(location)

    def serialize(self) -> bytes:
        """
        Serialize keys into bytes.

        Serialized keys are not encrypted, so be careful how you store/transfer them!
        `serialize_to_file` is supposed to be more performant as it avoid copying the buffer
        between the Compiler and the Frontend.

        Returns:
            bytes:
                serialized keys
        """

        if self._keyset is None:
            message = "Keys cannot be serialized before they are generated"
            raise RuntimeError(message)

        serialized_keyset = self._keyset.serialize()
        return serialized_keyset

    def serialize_to_file(self, path: Path):
        """
        Serialize keys into a file.

        Serialized keys are not encrypted, so be careful how you store/transfer them!
        This is supposed to be more performant than `serialize` as it avoid copying the buffer
        between the Compiler and the Frontend.

        Args:
            path (Path): where to save serialized keys
        """
        if self._keyset is None:
            message = "Keys cannot be serialized before they are generated"
            raise RuntimeError(message)

        self._keyset.serialize_to_file(str(path))

    @staticmethod
    def deserialize(serialized_keys: Union[Path, bytes]) -> "Keys":
        """
        Deserialize keys from file or buffer.

        Prefer using a Path instead of bytes in case of big Keys. It reduces memory usage.

        Args:
            serialized_keys (Union[Path, bytes]):
                previously serialized keys (either Path or buffer)

        Returns:
            Keys:
                deserialized keys
        """

        keyset = None
        if isinstance(serialized_keys, Path):
            keyset = Keyset.deserialize_from_file(str(serialized_keys))
        elif isinstance(serialized_keys, bytes):
            keyset = Keyset.deserialize(serialized_keys)
        assert keyset is not None, "serialized_keys should be either Path or bytes"

        # pylint: disable=protected-access
        result = Keys(None)
        result._keyset = keyset
        # pylint: enable=protected-access

        return result

    @property
    def specs(self) -> Optional[ClientSpecs]:
        """
        Return the associated client specs if any.
        """
        return self._specs  # pragma: no cover

    @property
    def evaluation(self) -> EvaluationKeys:
        """
        Get only evaluation keys.
        """

        self.generate(force=False)
        assert self._keyset is not None
        return EvaluationKeys(self._keyset.get_server_keys())
