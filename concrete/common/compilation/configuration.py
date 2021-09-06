"""Module for compilation configuration."""


class CompilationConfiguration:
    """Class that allows the compilation process to be customized."""

    dump_artifacts_on_unexpected_failures: bool
    enable_topological_optimizations: bool

    def __init__(
        self,
        dump_artifacts_on_unexpected_failures: bool = True,
        enable_topological_optimizations: bool = True,
    ):
        self.dump_artifacts_on_unexpected_failures = dump_artifacts_on_unexpected_failures
        self.enable_topological_optimizations = enable_topological_optimizations
