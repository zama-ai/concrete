"""Module for compilation configuration."""


class CompilationConfiguration:
    """Class that allows the compilation process to be customized."""

    dump_artifacts_on_unexpected_failures: bool
    enable_topological_optimizations: bool
    check_every_input_in_inputset: bool
    treat_warnings_as_errors: bool
    enable_unsafe_features: bool
    random_inputset_samples: int
    use_insecure_key_cache: bool

    def __init__(
        self,
        dump_artifacts_on_unexpected_failures: bool = True,
        enable_topological_optimizations: bool = True,
        check_every_input_in_inputset: bool = False,
        treat_warnings_as_errors: bool = False,
        enable_unsafe_features: bool = False,
        random_inputset_samples: int = 30,
        use_insecure_key_cache: bool = False,
    ):
        self.dump_artifacts_on_unexpected_failures = dump_artifacts_on_unexpected_failures
        self.enable_topological_optimizations = enable_topological_optimizations
        self.check_every_input_in_inputset = check_every_input_in_inputset
        self.treat_warnings_as_errors = treat_warnings_as_errors
        self.enable_unsafe_features = enable_unsafe_features
        self.random_inputset_samples = random_inputset_samples
        self.use_insecure_key_cache = use_insecure_key_cache

    def __eq__(self, other) -> bool:
        return isinstance(other, CompilationConfiguration) and self.__dict__ == other.__dict__
