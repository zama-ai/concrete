# bench: Unit Target: Multi Table Lookup

import math

import numpy as np
from common import BENCHMARK_CONFIGURATION

import concrete.numpy as hnp


def main():
    input_bits = 3

    square_table = hnp.LookupTable([i ** 2 for i in range(2 ** input_bits)])
    sqrt_table = hnp.LookupTable([int(math.sqrt(i)) for i in range(2 ** input_bits)])
    multi_table = hnp.MultiLookupTable(
        [
            [square_table, sqrt_table],
            [square_table, sqrt_table],
            [square_table, sqrt_table],
        ]
    )

    def function_to_compile(x):
        return multi_table[x]

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(input_bits), shape=(3, 2))

    # bench: Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
        [(np.random.randint(0, 2 ** input_bits, size=(3, 2)),) for _ in range(32)],
        compilation_configuration=BENCHMARK_CONFIGURATION,
    )
    # bench: Measure: End

    inputs = []
    labels = []
    for _ in range(50):
        sample_x = np.random.randint(0, 2 ** input_bits, size=(3, 2))

        inputs.append([sample_x])
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
