# Target: Single Table Lookup

import random

import concrete.numpy as hnp


def main():
    input_bits = 3

    entries = [i ** 2 for i in range(2 ** input_bits)]
    table = hnp.LookupTable(entries)

    def function_to_compile(x):
        return table[x]

    x = hnp.EncryptedScalar(hnp.UnsignedInteger(input_bits))

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
        [(i,) for i in range(2 ** input_bits)],
    )
    # Measure: End

    inputs = []
    labels = []
    for _ in range(10):
        sample_x = random.randint(0, (2 ** input_bits) - 1)

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
