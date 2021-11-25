# bench: Unit Target: np.negative(x)

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        # Such that the result is positive
        return 10 * np.ones(shape=(10, 6), dtype=np.uint8) + np.negative(x)

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(3), shape=(10, 6))

    inputset = [(np.random.randint(0, 2 ** 3, size=(10, 6)),) for _ in range(128)]

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(0, 2 ** 3, size=(10, 6), dtype=np.uint8)

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
    # bench: Alert: Accuracy (%) < 95


if __name__ == "__main__":
    main()
