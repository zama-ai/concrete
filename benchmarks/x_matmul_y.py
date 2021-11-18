# bench: Unit Target: np.matmul(x, y)

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    def function_to_compile(x, y):
        return np.matmul(x, y)

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(2), shape=(2, 3))
    y = hnp.EncryptedTensor(hnp.UnsignedInteger(2), shape=(3, 2))

    inputset = [
        (np.random.randint(0, 2 ** 2, size=(2, 3)), np.random.randint(0, 2 ** 2, size=(3, 2)))
        for _ in range(128)
    ]

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(0, 2 ** 2, size=(2, 3), dtype=np.uint8)
        sample_y = np.random.randint(0, 2 ** 2, size=(3, 2), dtype=np.uint8)

        inputs.append([sample_x, sample_y])
        labels.append(function_to_compile(*inputs[-1]))

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x, "y": y},
        inputset,
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # bench: Measure: End

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
