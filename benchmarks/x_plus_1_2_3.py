# Target: x + [1, 2, 3]

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        return x + np.array([1, 2, 3])

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(3), shape=(3,))

    inputset = [(np.random.randint(0, 2 ** 3, size=(3,)),) for _ in range(32)]

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
        inputset,
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(0, 2 ** 3, size=(3,))

        inputs.append([sample_x])
        labels.append(function_to_compile(*inputs[-1]))

    correct = 0
    for input_i, label_i in zip(inputs, labels):
        # Measure: Evaluation Time (ms)
        result_i = engine.run(*input_i)
        # Measure: End

        if result_i == label_i:
            correct += 1

    # Measure: Accuracy (%) = (correct / len(inputs)) * 100


if __name__ == "__main__":
    main()
