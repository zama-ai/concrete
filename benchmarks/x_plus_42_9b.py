# bench: Unit Target: x + 42 (9b)

import random

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():

    max_precision = 9

    def function_to_compile(x):
        return x + 42

    x = hnp.EncryptedScalar(hnp.UnsignedInteger(max_precision))

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
        [(i,) for i in range(2 ** max_precision - 42)],
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # bench: Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2 ** max_precision - 1 - 42)

        inputs.append([sample_x])
        labels.append(function_to_compile(*inputs[-1]))

    correct = 0
    for input_i, label_i in zip(inputs, labels):
        # bench: Measure: Evaluation Time (ms)
        result_i = engine.run(*input_i)
        # bench: Measure: End

        if np.array_equal(result_i, label_i):
            correct += 1

    # bench: Measure: Accuracy (%) = (correct / len(inputs)) * 100
    # bench: Alert: Accuracy (%) != 100


if __name__ == "__main__":
    main()
