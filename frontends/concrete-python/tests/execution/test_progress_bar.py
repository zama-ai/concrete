"""
Tests of execution of the progress bar.
"""
from concrete import fhe


def test_progress_bar(helpers, monkeypatch):
    """
    Test progress bar.
    """
    monkeypatch.setattr(
        "concrete.fhe.mlir.converter.Converter.stdout_with_ansi_support", lambda: True
    )

    def function(x):
        for _ in range(200):
            x += x
        return x

    configuration = helpers.configuration().fork(show_fhe_execution_progress=True)
    compiler = fhe.Compiler(function, {"x": "encrypted"})
    inputset = [0, 4]
    circuit = compiler.compile(inputset, configuration)

    expecteds = [("Fhe:   {i}% |", i // 50) for i in range(101)]
    for line in circuit.mlir.splitlines():
        if "FHE.add_eint" in line:
            next_is_tracing = True
        elif "Tracing.trace_message" in line:
            assert next_is_tracing
            expected = expecteds.pop(0)
            msg = line.split("msg = ")[1].split('"')[1]
            assert expected[0] in msg
            assert "." * expected[1] in msg
            next_is_tracing = False


def test_progress_bar_no_ansi(helpers):
    """
    Test progress bar.
    """

    def function(x):
        for _ in range(100):
            x += x
        return x

    configuration = helpers.configuration().fork(show_fhe_execution_progress=True)
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [0, 4]
    circuit = compiler.compile(inputset, configuration)
    print(circuit.mlir)
    next_is_tracing = False
    for line in circuit.mlir.splitlines():
        print(line)
        if "FHE.add_eint" in line:
            next_is_tracing = True
        elif "Tracing.trace_message" in line:
            if "______" in line:
                continue
            assert next_is_tracing
            msg = line.split("msg = ")[1].split('"')[1]
            assert msg.count("\\E2\\96\\88") == 1  # add one stone
            next_is_tracing = False
    assert "Finished" in msg
