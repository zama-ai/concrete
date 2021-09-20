import random

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        return x + 42

    n_bits = 3
    x = hnp.EncryptedScalar(hnp.UnsignedInteger(n_bits))

    engine = hnp.compile_numpy_function(
        function_to_compile,
        {"x": x},
        [(i,) for i in range(2 ** n_bits)],
    )

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2 ** n_bits - 1)

        inputs.append([sample_x])
        labels.append(function_to_compile(*inputs[-1]))

    correct = 0
    for input_i, label_i in zip(inputs, labels):
        result_i = engine.run(*input_i)

        if result_i == label_i:
            correct += 1


if __name__ == "__main__":
    main()
