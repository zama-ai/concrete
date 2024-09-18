# Benchmarking

This document gives an overview of the benchmarking infrastructure of Concrete.

## Concrete Python

Concrete Python uses [progress-tracker-python](https://github.com/zama-ai/progress-tracker-python) to do benchmarks. Please refer to its [README](https://github.com/zama-ai/progress-tracker-python/blob/main/README.md) to learn how it works.

### How to run all benchmarks?

Use the makefile target:

```shell
make benchmark
```

Note that this command removes the previous benchmark results before doing the benchmark.

### How to run a single benchmark?

Since the full benchmark suite takes a long time to run, it's not recommended for development. Instead, use the following command to run just a single benchmark.

```shell
TARGET=foo make benchmark-target
```

This command would only run the benchmarks defined in `benchmarks/foo.py`. It also retains the previous runs, so it can be run back to back to collect data from multiple benchmarks.

### How to add new benchmarks?

Simply add a new Python script in `benchmarks` directory and write your logic. 

The recommended file structure is as follows:

```python
# import progress tracker
import py_progress_tracker as progress

# import any other dependencies
from concrete import fhe

# create a list of targets to benchmark
targets = [
    {
        "id": (
            f"name-of-the-benchmark :: "
            f"parameter1 = {foo} | parameter2 = {bar}"
        ),
        "name": (
            f"Name of the benchmark with parameter1 of {foo} and parameter2 of {bar}"
        ),
        "parameters": {
            "parameter1": foo,
            "parameter2": bar,
        },
    }
]

# write the benchmark logic
@progress.track(targets)
def main(parameter1, parameter2):
    ...

    # to track timings
    with progress.measure(id="some-metric-ms", label="Some metric (ms)"):
        # execution time of this block will be measured
        ...

    ...

    # to track values
    progress.measure(id="another-metric", label="Another metric", value=some_metric)

    ...
```

Feel free to check `benchmarks/primitive.py` to see this structure in action.
