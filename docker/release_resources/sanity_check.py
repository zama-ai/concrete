import random

# Check that extras are installed in the docker image
import pygraphviz

print("Extras import check OK")

import concrete.numpy as hnp


def main():
    def function_to_compile(x):
        return x + 42

    n_bits = 3

    compiler = hnp.NPFHECompiler(
        function_to_compile,
        {"x": "encrypted"},
    )

    print("Compiling...")

    engine = compiler.compile_on_inputset(range(2 ** n_bits))

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2 ** n_bits - 1)

        inputs.append([sample_x])
        labels.append(function_to_compile(*inputs[-1]))

    correct = 0
    for idx, (input_i, label_i) in enumerate(zip(inputs, labels), 1):
        print(f"Inference #{idx}")
        result_i = engine.run(*input_i)

        if result_i == label_i:
            correct += 1

    print(f"{correct}/{len(inputs)}")


if __name__ == "__main__":
    main()
