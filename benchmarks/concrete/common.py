import concrete.numpy as hnp
from concrete.numpy import compile as compile_

# This is only for benchmarks to speed up compilation times
# pylint: disable=protected-access
compile_._COMPILE_FHE_INSECURE_KEY_CACHE_DIR = "/tmp/keycache"
# pylint: enable=protected-access

BENCHMARK_CONFIGURATION = hnp.CompilationConfiguration(
    check_every_input_in_inputset=True,
    dump_artifacts_on_unexpected_failures=True,
    enable_topological_optimizations=True,
    enable_unsafe_features=True,
    treat_warnings_as_errors=True,
    use_insecure_key_cache=True,
)
