# Target: x**2

import random

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        return x ** 2

    x = hnp.EncryptedScalar(hnp.UnsignedInteger(3))

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
        iter([(6,), (1,), (5,), (2,)]),
    )
    # Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2 ** 3 - 1)

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
