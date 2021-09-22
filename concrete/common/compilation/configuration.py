"""Module for compilation configuration."""


class CompilationConfiguration:
    """Class that allows the compilation process to be customized."""

    dump_artifacts_on_unexpected_failures: bool
    enable_topological_optimizations: bool
    check_every_input_in_inputset: bool

    def __init__(
        self,
        dump_artifacts_on_unexpected_failures: bool = True,
        enable_topological_optimizations: bool = True,
        check_every_input_in_inputset: bool = False,
    ):
        self.dump_artifacts_on_unexpected_failures = dump_artifacts_on_unexpected_failures
        self.enable_topological_optimizations = enable_topological_optimizations
        self.check_every_input_in_inputset = check_every_input_in_inputset
