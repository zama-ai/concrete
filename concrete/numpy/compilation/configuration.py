"""
Declaration of `Configuration` class.
"""

from typing import Optional

_INSECURE_KEY_CACHE_LOCATION: Optional[str] = None


class Configuration:
    """
    Configuration class, to allow the compilation process to be customized.
    """

    dump_artifacts_on_unexpected_failures: bool
    enable_unsafe_features: bool
    use_insecure_key_cache: bool
    loop_parallelize: bool
    dataflow_parallelize: bool
    auto_parallelize: bool

    def __init__(
        self,
        dump_artifacts_on_unexpected_failures: bool = True,
        enable_unsafe_features: bool = False,
        use_insecure_key_cache: bool = False,
        loop_parallelize: bool = True,
        dataflow_parallelize: bool = False,
        auto_parallelize: bool = False,
    ):
        self.dump_artifacts_on_unexpected_failures = dump_artifacts_on_unexpected_failures
        self.enable_unsafe_features = enable_unsafe_features
        self.use_insecure_key_cache = use_insecure_key_cache
        self.loop_parallelize = loop_parallelize
        self.dataflow_parallelize = dataflow_parallelize
        self.auto_parallelize = auto_parallelize

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
