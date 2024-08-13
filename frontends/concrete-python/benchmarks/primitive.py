"""
Benchmark primitive operations.
"""

# pylint: disable=import-error,cell-var-from-loop,redefined-outer-name

import random

import py_progress_tracker as progress

from concrete import fhe

targets = []
configuration = fhe.Configuration()

# Table Lookup
for bit_width in range(2, 8 + 1):
    targets.append(
        {
            "id": f"table-lookup :: tlu[eint{bit_width}]",
            "name": f"{bit_width}-bit table lookup",
            "parameters": {
                "function": lambda x: x // 2,
                "encryption": {"x": "encrypted"},
                "inputset": fhe.inputset(lambda _: random.randint(0, (2**bit_width) - 1)),
                "configuration": configuration,
            },
        }
    )

# Encrypted Multiplication
for bit_width in range(2, 8 + 1):
    targets.append(
        {
            "id": f"encrypted-multiplication :: eint{bit_width} * eint{bit_width}",
            "name": f"{bit_width}-bit encrypted multiplication",
            "parameters": {
                "function": lambda x, y: x * y,
                "encryption": {"x": "encrypted", "y": "encrypted"},
                "inputset": fhe.inputset(
                    lambda _: random.randint(0, (2**bit_width) - 1),
                    lambda _: random.randint(0, (2**bit_width) - 1),
                ),
                "configuration": configuration,
            },
        }
    )


@progress.track(targets)
def main(function, encryption, inputset, configuration):
    """
    Benchmark a target.

    Args:
        function:
            function to benchmark

        encryption:
            encryption status of the arguments of the function

        inputset:
            inputset to use for compiling the function

        configuration:
            configuration to use for compilation
    """

    compiler = fhe.Compiler(function, encryption)

    print("Compiling...")
    with progress.measure(id="compilation-time-ms", label="Compilation Time (ms)"):
        circuit = compiler.compile(inputset, configuration)

    progress.measure(
        id="complexity",
        label="Complexity",
        value=circuit.complexity,
    )

    print("Generating keys...")
    with progress.measure(id="key-generation-time-ms", label="Key Generation Time (ms)"):
        circuit.keygen(force=True)

    progress.measure(
        id="evaluation-key-size-mb",
        label="Evaluation Key Size (MB)",
        value=(len(circuit.keys.evaluation.serialize()) / (1024 * 1024)),
    )

    # pylint: disable=unused-variable

    print("Warming up...")
    sample = random.choice(inputset)
    encrypted = circuit.encrypt(*sample)
    ran = circuit.run(encrypted)
    decrypted = circuit.decrypt(ran)  # noqa: F841

    # pylint: enable=unused-variable

    def calculate_input_output_size(input_output):
        if isinstance(input_output, tuple):
            result = sum(len(value.serialize()) for value in input_output)
        else:
            result = len(input_output.serialize())
        return result / (1024 * 1024)

    progress.measure(
        id="input-ciphertext-size-mb",
        label="Input Ciphertext Size (MB)",
        value=calculate_input_output_size(encrypted),
    )
    progress.measure(
        id="output-ciphertext-size-mb",
        label="Output Ciphertext Size (MB)",
        value=calculate_input_output_size(ran),
    )

    for i in range(10):
        print(f"Running subsample {i + 1} out of 10...")

        sample = random.choice(inputset)
        with progress.measure(id="encryption-time-ms", label="Encryption Time (ms)"):
            encrypted = circuit.encrypt(*sample)
        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            ran = circuit.run(encrypted)
        with progress.measure(id="decryption-time-ms", label="Decryption Time (ms)"):
            output = circuit.decrypt(ran)

        progress.measure(
            id="accuracy",
            label="Accuracy",
            value=int(output == function(*sample)),
        )
