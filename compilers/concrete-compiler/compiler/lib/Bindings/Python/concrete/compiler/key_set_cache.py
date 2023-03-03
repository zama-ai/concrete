#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""KeySetCache.

Cache for keys to avoid generating similar keys multiple times.
"""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    KeySetCache as _KeySetCache,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class KeySetCache(WrapperCpp):
    """KeySetCache is a cache for KeySet to avoid generating similar keys multiple times.

    Keys get cached and can be later used instead of generating a new keyset which can take a lot of time.
    """

    def __init__(self, keyset_cache: _KeySetCache):
        """Wrap the native Cpp object.

        Args:
            keyset_cache (_KeySetCache): object to wrap

        Raises:
            TypeError: if keyset_cache is not of type _KeySetCache
        """
        if not isinstance(keyset_cache, _KeySetCache):
            raise TypeError(
                f"key_set_cache must be of type _KeySetCache, not {type(keyset_cache)}"
            )
        super().__init__(keyset_cache)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(cache_path: str) -> "KeySetCache":
        """Build a KeySetCache located at cache_path.

        Args:
            cache_path (str): path to the cache

        Raises:
            TypeError: if the path is not of type str.

        Returns:
            KeySetCache
        """
        if not isinstance(cache_path, str):
            raise TypeError(
                f"cache_path must to be of type str, not {type(cache_path)}"
            )
        return KeySetCache.wrap(_KeySetCache(cache_path))

    # pylint: enable=arguments-differ
