# bench: Unit Target: x[y, z] (Encrypted)

import random

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    def function_to_compile(x, y, z):
        return x[y, z]

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(3), shape=(4, 2))
    y = hnp.EncryptedScalar(hnp.UnsignedInteger(2))
    z = hnp.EncryptedScalar(hnp.UnsignedInteger(1))

    inputset = [
        (
            np.random.randint(0, 2 ** 3, size=(4, 2)),
            random.randint(0, (2 ** 2) - 1),
            random.randint(0, (2 ** 1) - 1),
        )
        for _ in range(32)
    ]

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x, "y": y, "z": z},
        inputset,
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # bench: Measure: End

    inputs = []
    labels = []
    for _ in range(100):
        sample_x = np.random.randint(0, 2 ** 3, size=(4, 2))
        sample_y = random.randint(0, (2 ** 2) - 1)
        sample_z = random.randint(0, (2 ** 1) - 1)

        inputs.append([sample_x, sample_y, sample_z])
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
