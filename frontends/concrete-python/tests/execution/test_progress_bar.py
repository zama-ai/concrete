"""
Tests of execution of the progress bar.
"""
from collections import Counter

from concrete import fhe


def test_progress_bar(helpers):
    """
    Test progress bar.
    """

    def function(x):
        for _ in range(3):
            x += x
        return x

    configuration = helpers.configuration().fork(show_fhe_execution_progress=True)
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [0, 4]
    circuit = compiler.compile(inputset, configuration)
    expecteds = [
        ("Fhe:   0% |", 50),
        ("Fhe:  33% |", 50 - 16),
        ("Fhe:  66% |", 50 - 33),
        ("Fhe: 100% |", 0),
    ]
    for line in circuit.mlir.splitlines():
        print(line)
        if "FHE.add_eint" in line:
            expecteds.pop(0)
        elif "Tracing.trace_message" in line:
            expected = expecteds[0]
            msg = line.split("msg = ")[1].split('"')[1]
            assert expected[0] in msg
            assert "." * expected[1] in msg
