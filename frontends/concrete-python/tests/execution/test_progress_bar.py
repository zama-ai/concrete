"""
Tests of execution of the progress bar.
"""
from concrete import fhe

STEP_ESCAPED = "\\E2\\96\\88"


def test_progress_bar(helpers, monkeypatch):
    """
    Test progress bar with interactive terminal.
    """
    monkeypatch.setattr(
        "concrete.fhe.mlir.converter.Converter.stdout_with_ansi_support", lambda: True
    )

    def function(x):
        acc = x
        for _ in range(200):
            acc += x
        return acc

    configuration = helpers.configuration().fork(show_fhe_execution_progress=True)
    compiler = fhe.Compiler(function, {"x": "encrypted"})
    inputset = [0, 4]
    circuit = compiler.compile(inputset, configuration)
    next_is_tracing = True
    expecteds = [(f" {i:>3}% ", i // 2) for i in range(101)]
    for line in circuit.mlir.splitlines():
        if "FHE.add_eint" in line:
            next_is_tracing = True
        elif "Tracing.trace_message" in line:
            assert next_is_tracing
            expected = expecteds.pop(0)
            msg = line.split("msg = ")[1].split('"')[1]
            assert expected[0] in msg
            assert msg.count(STEP_ESCAPED) == expected[1]
            next_is_tracing = False


def test_progress_bar_no_ansi(helpers):
    """
    Test progress bar with non interactive terminal (pipe, notebook, etc).
    """

    def function(x):
        acc = x
        for _ in range(100):
            acc += x
        return acc

    configuration = helpers.configuration().fork(show_fhe_execution_progress=True)
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [0, 4]
    circuit = compiler.compile(inputset, configuration)
    next_is_tracing = True
    for line in circuit.mlir.splitlines():
        print(line)
        if "FHE.add_eint" in line:
            next_is_tracing = True
        elif "Tracing.trace_message" in line:
            if "______" in line:
                continue
            assert next_is_tracing
            msg = line.split("msg = ")[1].split('"')[1]
            assert msg.count(STEP_ESCAPED) == 1  # add one stone
            next_is_tracing = False
    assert "Finished" in msg


def test_progress_bar_empty_circuit(helpers):
    """
    Test progress bar on empty circuit.
    """

    def function(x):
        return x

    configuration = helpers.configuration().fork(show_fhe_execution_progress=True)
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [0, 4]
    circuit = compiler.compile(inputset, configuration)
    assert "Tracing.trace_message" not in circuit.mlir
