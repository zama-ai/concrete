"""
Declaration of `Configuration` class.
"""

import platform
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Optional, Union, get_type_hints

from .utils import friendly_type_format

DEFAULT_P_ERROR = None
DEFAULT_GLOBAL_P_ERROR = 1 / 100_000


class ParameterSelectionStrategy(str, Enum):
    """
    ParameterSelectionStrategy, to set optimization strategy.
    """

    V0 = "v0"
    MONO = "mono"
    MULTI = "multi"

    @classmethod
    def parse(cls, string: str) -> "ParameterSelectionStrategy":
        """Convert a string to a ParameterSelectionStrategy."""
        if isinstance(string, cls):
            return string
        if not isinstance(string, str):
            message = f"{string} cannot be parsed to a {cls.__name__}"
            raise TypeError(message)
        for value in ParameterSelectionStrategy:
            if string.lower() == value.value:
                return value
        message = (
            f"'{string}' is not a valid '{friendly_type_format(cls)}' ("
            f"{', '.join(v.value for v in ParameterSelectionStrategy)})"
        )
        raise ValueError(message)


class Configuration:
    """
    Configuration class, to allow the compilation process to be customized.
    """

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
    single_precision: bool
    parameter_selection_strategy: Union[ParameterSelectionStrategy, str]
    show_progress: bool
    progress_title: str
    progress_tag: Union[bool, int]

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
        single_precision: bool = True,
        parameter_selection_strategy: Union[
            ParameterSelectionStrategy, str
        ] = ParameterSelectionStrategy.MONO,
        show_progress: bool = False,
        progress_title: str = "",
        progress_tag: Union[bool, int] = False,
    ):  # pylint: disable=too-many-arguments
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
        self.single_precision = single_precision
        self.parameter_selection_strategy = parameter_selection_strategy
        self.show_progress = show_progress
        self.progress_title = progress_title
        self.progress_tag = progress_tag

        self._validate()

    class Keep:
        """Keep previous arg value during fork."""

    KEEP = Keep()

    def fork(
        self,
        /,
        # pylint: disable=unused-argument
        verbose: Union[Keep, bool] = KEEP,
        show_graph: Union[Keep, Optional[bool]] = KEEP,
        show_mlir: Union[Keep, Optional[bool]] = KEEP,
        show_optimizer: Union[Keep, Optional[bool]] = KEEP,
        dump_artifacts_on_unexpected_failures: Union[Keep, bool] = KEEP,
        enable_unsafe_features: Union[Keep, bool] = KEEP,
        use_insecure_key_cache: Union[Keep, bool] = KEEP,
        insecure_key_cache_location: Union[Keep, Optional[Union[Path, str]]] = KEEP,
        loop_parallelize: Union[Keep, bool] = KEEP,
        dataflow_parallelize: Union[Keep, bool] = KEEP,
        auto_parallelize: Union[Keep, bool] = KEEP,
        jit: Union[Keep, bool] = KEEP,
        p_error: Union[Keep, Optional[float]] = KEEP,
        global_p_error: Union[Keep, Optional[float]] = KEEP,
        auto_adjust_rounders: Union[Keep, bool] = KEEP,
        single_precision: Union[Keep, bool] = KEEP,
        parameter_selection_strategy: Union[Keep, Union[ParameterSelectionStrategy, str]] = KEEP,
        show_progress: Union[Keep, bool] = KEEP,
        progress_title: Union[Keep, str] = KEEP,
        progress_tag: Union[Keep, Union[bool, int]] = KEEP,
    ) -> "Configuration":
        """
        Get a new configuration from another one specified changes.

        See Configuration.

        """
        args = locals()
        result = deepcopy(self)
        for name in get_type_hints(Configuration.__init__):
            value = args[name]
            if isinstance(value, Configuration.Keep):
                continue
            setattr(result, name, value)

        # pylint: disable=protected-access
        result._validate()
        # pylint: enable=protected-access

        return result

    def _validate(self):
        """
        Validate configuration.
        """
        for name, hint in get_type_hints(Configuration.__init__).items():
            original_hint = hint
            value = getattr(self, name)
            if name == "parameter_selection_strategy":
                try:
                    value = ParameterSelectionStrategy.parse(value)
                except ValueError as exc:
                    message = f"Unexpected value for keyword argument '{name}', {str(exc)}"
                    # pylint: disable=raise-missing-from
                    raise ValueError(message)  # noqa: B904
                except TypeError:
                    pass  # handle by the generic check
            if str(hint).startswith("typing.Union") or str(hint).startswith("typing.Optional"):
                if isinstance(value, tuple(hint.__args__)):
                    continue
            elif isinstance(value, hint):
                continue
            hint = friendly_type_format(original_hint)
            value_type = friendly_type_format(type(value))
            message = (
                f"Unexpected type for keyword argument '{name}' "
                f"(expected '{hint}', got '{value_type}')"
            )
            raise TypeError(message)

        if not self.enable_unsafe_features:  # noqa: SIM102
            if self.use_insecure_key_cache:
                message = "Insecure key cache cannot be used without enabling unsafe features"
                raise RuntimeError(message)

        if self.use_insecure_key_cache and self.insecure_key_cache_location is None:
            message = "Insecure key cache cannot be enabled without specifying its location"
            raise RuntimeError(message)

        if platform.system() == "Darwin" and self.dataflow_parallelize:  # pragma: no cover
            message = "Dataflow parallelism is not available in macOS"
            raise RuntimeError(message)


def __check_fork_consistency():
    hints_init = get_type_hints(Configuration.__init__)
    hints_fork = get_type_hints(Configuration.fork)
    diff = set.symmetric_difference(set(hints_init), set(hints_fork) - {"return"})
    if diff:  # pragma: no cover
        message = f"Configuration.fork is inconsistent with Configuration for: {diff}"
        raise TypeError(message)
    for name, init_hint in hints_init.items():
        fork_hint = hints_fork[name]
        if Union[Configuration.Keep, init_hint] != fork_hint:  # pragma: no cover
            fork_hint = friendly_type_format(fork_hint)
            init_hint = friendly_type_format(init_hint)
            message = (
                f"Configuration.fork parameter {name}: {fork_hint} is inconsistent"
                f"with Configuration type: {init_hint}"
            )
            raise TypeError(message)


__check_fork_consistency()
