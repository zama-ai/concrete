# Target: x * [1, 2, 3] (Broadcasted)

import numpy as np

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        return x * np.array([1, 2, 3])

    x = hnp.EncryptedTensor(hnp.UnsignedInteger(3), shape=(2, 3))

    inputset = [
        (np.array([[0, 7, 7], [6, 2, 4]]),),
        (np.array([[6, 2, 4], [1, 3, 1]]),),
        (np.array([[6, 2, 4], [5, 1, 2]]),),
        (np.array([[5, 1, 2], [0, 7, 7]]),),
    ]

    # Measure: Compilation Time (ms)
    engine = hnp.compile_numpy_function(function_to_compile, {"x": x}, inputset)
    # Measure: End

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = np.random.randint(0, 2 ** 3, size=(2, 3))

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
