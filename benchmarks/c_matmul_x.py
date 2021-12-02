# bench: Unit Target: np.matmul(c, x)

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    c = np.arange(1, 7).reshape((3, 2))

    def function_to_compile(x):
        return np.matmul(c, x)

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(2), shape=(2, 3))

    inputset = [np.random.randint(0, 2 ** 2, size=(2, 3)) for _ in range(128)]

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(0, 2 ** 2, size=(2, 3), dtype=np.uint8)

        inputs.append([sample_x])
        labels.append(function_to_compile(*inputs[-1]))

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
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
