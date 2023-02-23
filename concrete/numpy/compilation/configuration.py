"""
Declaration of `Configuration` class.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, get_type_hints

DEFAULT_P_ERROR = None
DEFAULT_GLOBAL_P_ERROR = 1 / 100_000


class Configuration:
    """
    Configuration class, to allow the compilation process to be customized.
    """

    # pylint: disable=too-many-instance-attributes

    verbose: bool
    show_graph: Optional[bool]
    show_mlir: Optional[bool]
    show_optimizer: Optional[bool]
    dump_artifacts_on_unexpected_failures: bool
    enable_unsafe_features: bool
    use_insecure_key_cache: bool
    loop_parallelize: bool
    dataflow_parallelize: bool
    auto_parallelize: bool
    jit: bool
    p_error: Optional[float]
    global_p_error: Optional[float]
    insecure_key_cache_location: Optional[str]
    auto_adjust_rounders: bool

    # pylint: enable=too-many-instance-attributes

    def _validate(self):
        """
        Validate configuration.
        """

        if not self.enable_unsafe_features:

            if self.use_insecure_key_cache:
                message = "Insecure key cache cannot be used without enabling unsafe features"
                raise RuntimeError(message)

        if self.use_insecure_key_cache and self.insecure_key_cache_location is None:
            message = "Insecure key cache cannot be enabled without specifying its location"
            raise RuntimeError(message)

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        verbose: bool = False,
        show_graph: Optional[bool] = None,
        show_mlir: Optional[bool] = None,
        show_optimizer: Optional[bool] = None,
        dump_artifacts_on_unexpected_failures: bool = True,
        enable_unsafe_features: bool = False,
        use_insecure_key_cache: bool = False,
        insecure_key_cache_location: Optional[Union[Path, str]] = None,
        loop_parallelize: bool = True,
        dataflow_parallelize: bool = False,
        auto_parallelize: bool = False,
        jit: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        auto_adjust_rounders: bool = False,
    ):
        self.verbose = verbose
        self.show_graph = show_graph
        self.show_mlir = show_mlir
        self.show_optimizer = show_optimizer
        self.dump_artifacts_on_unexpected_failures = dump_artifacts_on_unexpected_failures
        self.enable_unsafe_features = enable_unsafe_features
        self.use_insecure_key_cache = use_insecure_key_cache
        self.insecure_key_cache_location = (
            str(insecure_key_cache_location) if insecure_key_cache_location is not None else None
        )
        self.loop_parallelize = loop_parallelize
        self.dataflow_parallelize = dataflow_parallelize
        self.auto_parallelize = auto_parallelize
        self.jit = jit
        self.p_error = p_error
        self.global_p_error = global_p_error
        self.auto_adjust_rounders = auto_adjust_rounders

        self._validate()

    # pylint: enable=too-many-arguments

    def fork(self, **kwargs) -> "Configuration":
        """
        Get a new configuration from another one specified changes.

        Args:
            **kwargs:
                changes to make

        Returns:
            Configuration:
                configuration that is forked from self and updated using kwargs
        """

        # pylint: disable=too-many-branches

        result = deepcopy(self)

        hints = get_type_hints(Configuration)
        for name, value in kwargs.items():
            if name not in hints:
                message = f"Unexpected keyword argument '{name}'"
                raise TypeError(message)

            hint = hints[name]
            expected = None
            is_correctly_typed = True

            if name == "insecure_key_cache_location":
                if not (value is None or isinstance(value, str)):
                    is_correctly_typed = False
                    expected = "Optional[str]"

            elif name == "p_error":
                if not (value is None or isinstance(value, float)):
                    is_correctly_typed = False
                    expected = "Optional[float]"

            elif name == "global_p_error":
                if not (value is None or isinstance(value, float)):
                    is_correctly_typed = False
                    expected = "Optional[float]"

            elif name in ["show_graph", "show_mlir", "show_optimizer"]:
                if not (value is None or isinstance(value, bool)):
                    is_correctly_typed = False
                    expected = "Optional[bool]"

            elif not isinstance(value, hint):  # type: ignore
                is_correctly_typed = False

            if not is_correctly_typed:
                if expected is None:
                    expected = hint.__name__ if hasattr(hint, "__name__") else str(hint)
                message = (
                    f"Unexpected type for keyword argument '{name}' "
                    f"(expected '{expected}', got '{type(value).__name__}')"
                )
                raise TypeError(message)

            setattr(result, name, value)

        # pylint: disable=protected-access
        result._validate()
        # pylint: enable=protected-access

        return result

        # pylint: enable=too-many-branches
