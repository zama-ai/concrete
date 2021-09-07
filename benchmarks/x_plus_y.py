# Target: x + y

import random

import concrete.numpy as hnp


def main():
    def function_to_compile(x, y):
        return x + y

    x = hnp.EncryptedScalar(hnp.UnsignedInteger(3))
    y = hnp.EncryptedScalar(hnp.UnsignedInteger(3))

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x, "y": y},
        iter([(6, 1), (1, 4), (5, 3), (2, 0), (7, 7)]),
    )
    # Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2 ** 3 - 1)
        sample_y = random.randint(0, 2 ** 3 - 1)

        inputs.append([sample_x, sample_y])
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
