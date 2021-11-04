# bench: Unit Target: x[y, 1:] (Clear)

import random

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    def function_to_compile(x, y):
        return x[y, 1:]

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(3), shape=(4, 5))
    y = hnp.ClearScalar(hnp.UnsignedInteger(2))

    inputset = [
        (np.random.randint(0, 2 ** 3, size=(4, 5)), random.randint(0, (2 ** 2) - 1))
        for _ in range(32)
    ]

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x, "y": y},
        inputset,
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # bench: Measure: End

    inputs = []
    labels = []
    for _ in range(100):
        sample_x = np.random.randint(0, 2 ** 3, size=(4, 5))
        sample_y = random.randint(0, (2 ** 2) - 1)

        inputs.append([sample_x, sample_y])
        labels.append(function_to_compile(*inputs[-1]))

    correct = 0
    for input_i, label_i in zip(inputs, labels):
        # bench: Measure: Evaluation Time (ms)
        result_i = engine.run(*input_i)
        # bench: Measure: End

        if result_i == label_i:
            correct += 1

    # bench: Measure: Accuracy (%) = (correct / len(inputs)) * 100
    # bench: Alert: Accuracy (%) != 100


if __name__ == "__main__":
    main()
