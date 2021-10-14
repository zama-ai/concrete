# bench: Unit Target: x + y

import random

from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    def function_to_compile(x, y):
        return x + y

    x = hnp.EncryptedScalar(hnp.UnsignedInteger(3))
    y = hnp.EncryptedScalar(hnp.UnsignedInteger(3))

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x, "y": y},
        [(random.randint(0, 7), random.randint(0, 7)) for _ in range(32)],
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # bench: Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2 ** 3 - 1)
        sample_y = random.randint(0, 2 ** 3 - 1)

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
