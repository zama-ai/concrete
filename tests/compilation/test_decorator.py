"""
Tests of `compiler` decorator.
"""

from concrete.numpy.compilation import DebugArtifacts, compiler


def test_call_compile(helpers):
    """
    Test `__call__` and `compile` methods of `compiler` decorator back to back.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"}, configuration=configuration)
    def function(x):
        return x + 42

    for i in range(10):
        function(i)

    circuit = function.compile()

    sample = 5
    helpers.check_execution(circuit, function, sample)


def test_compiler_verbose_trace(helpers, capsys):
    """
    Test `trace` method of `compiler` decorator with verbose flag.
    """

    configuration = helpers.configuration()
    artifacts = DebugArtifacts()

    @compiler({"x": "encrypted"}, configuration=configuration, artifacts=artifacts)
    def function(x):
        return x + 42

    inputset = range(10)
    function.trace(inputset, show_graph=True)

    captured = capsys.readouterr()
    assert captured.out.strip() == (
        f"""

Computation Graph
------------------------------------------------
{str(list(artifacts.textual_representations_of_graphs.values())[-1])}
------------------------------------------------

        """.strip()
    )


def test_compiler_verbose_compile(helpers, capsys):
    """
    Test `compile` method of `compiler` decorator with verbose flag.
    """

    configuration = helpers.configuration()
    artifacts = DebugArtifacts()

    @compiler({"x": "encrypted"}, configuration=configuration, artifacts=artifacts)
    def function(x):
        return x + 42

    inputset = range(10)
    function.compile(inputset, show_graph=True, show_mlir=True)

    captured = capsys.readouterr()
    assert captured.out.strip() == (
        f"""

Computation Graph
--------------------------------------------------------------------------------
{list(artifacts.textual_representations_of_graphs.values())[-1]}
--------------------------------------------------------------------------------

MLIR
--------------------------------------------------------------------------------
{artifacts.mlir_to_compile}
--------------------------------------------------------------------------------

        """.strip()
    )
