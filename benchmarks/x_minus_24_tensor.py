# Target: x - 24 (Tensor)

import numpy as np

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        return x - 24

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(6), shape=(3,))

    inputset = [
        (np.array([36, 50, 24]),),
        (np.array([41, 60, 51]),),
        (np.array([25, 31, 24]),),
        (np.array([34, 47, 27]),),
    ]

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(function_to_compile, {"x": x}, inputset)
    # Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(24, 2 ** 6, size=(3,))

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
