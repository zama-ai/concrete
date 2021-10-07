import concrete.numpy as hnp

BENCHMARK_CONFIGURATION = hnp.CompilationConfiguration(
    dump_artifacts_on_unexpected_failures=True,
    enable_topological_optimizations=True,
    check_every_input_in_inputset=True,
    treat_warnings_as_errors=True,
)
