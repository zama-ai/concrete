# bench: Unit Target: np.matmul(x, c)

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    c = np.arange(20).reshape((4, 5))

    def function_to_compile(x):
        return np.matmul(x, c)

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(3), shape=(2, 4))

    inputset = [(np.random.randint(0, 2 ** 3, size=(2, 4))) for _ in range(128)]

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(0, 2 ** 3, size=(2, 4))

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

        if result_i == label_i:
            correct += 1

    # bench: Measure: Accuracy (%) = (correct / len(inputs)) * 100
    # bench: Alert: Accuracy (%) != 100


if __name__ == "__main__":
    main()
