"""Module for compilation configuration."""


class CompilationConfiguration:
    """Class that allows the compilation process to be customized."""

    enable_topological_optimizations: bool

    def __init__(
        self,
        enable_topological_optimizations: bool = True,
    ):
        self.enable_topological_optimizations = enable_topological_optimizations
