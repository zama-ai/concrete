"""
Declaration of `CompilationConfiguration` class.
"""

from typing import Optional

_INSECURE_KEY_CACHE_LOCATION: Optional[str] = None


class CompilationConfiguration:
    """
    CompilationConfiguration class, to allow the compilation process to be customized.
    """

    dump_artifacts_on_unexpected_failures: bool
    enable_unsafe_features: bool
    use_insecure_key_cache: bool

    def __init__(
        self,
        dump_artifacts_on_unexpected_failures: bool = True,
        enable_unsafe_features: bool = False,
        use_insecure_key_cache: bool = False,
    ):
        self.dump_artifacts_on_unexpected_failures = dump_artifacts_on_unexpected_failures
        self.enable_unsafe_features = enable_unsafe_features
        self.use_insecure_key_cache = use_insecure_key_cache

        if not enable_unsafe_features and use_insecure_key_cache:
            raise RuntimeError("Insecure key cache cannot be used without enabling unsafe features")

    @staticmethod
    def insecure_key_cache_location() -> Optional[str]:
        """
        Get insecure key cache location.

        Returns:
            Optional[str]:
                insecure key cache location if configured, None otherwise
        """

        return _INSECURE_KEY_CACHE_LOCATION
