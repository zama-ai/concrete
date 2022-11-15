"""
Tests of `compiler` and `circuit` decorators.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


def test_compiler_call_and_compile(helpers):
    """
    Test `__call__` and `compile` methods of `compiler` decorator back to back.
    """

    configuration = helpers.configuration()

    @cnp.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    for i in range(10):
        function(i)

    circuit = function.compile(configuration=configuration)

    sample = 5
    helpers.check_execution(circuit, function, sample)


def test_compiler_verbose_trace(helpers, capsys):
    """
    Test `trace` method of `compiler` decorator with verbose flag.
    """

    configuration = helpers.configuration()
    artifacts = cnp.DebugArtifacts()

    @cnp.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    function.trace(inputset, configuration, artifacts, show_graph=True)

    captured = capsys.readouterr()
    assert captured.out.strip() == (
        f"""

Computation Graph
------------------------------------------------------------------
{str(list(artifacts.textual_representations_of_graphs.values())[-1][-1])}
------------------------------------------------------------------

        """.strip()
    )


def test_compiler_verbose_compile(helpers, capsys):
    """
    Test `compile` method of `compiler` decorator with verbose flag.
    """

    configuration = helpers.configuration()
    artifacts = cnp.DebugArtifacts()

    @cnp.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    function.compile(inputset, configuration, artifacts, verbose=True)

    captured = capsys.readouterr()
    assert captured.out.strip().startswith(
        f"""

Computation Graph
--------------------------------------------------------------------------------
{list(artifacts.textual_representations_of_graphs.values())[-1][-1]}
--------------------------------------------------------------------------------

MLIR
--------------------------------------------------------------------------------
{artifacts.mlir_to_compile}
--------------------------------------------------------------------------------

Optimizer
--------------------------------------------------------------------------------

        """.strip()
    )


def test_compiler_verbose_virtual_compile(helpers, capsys):
    """
    Test `compile` method of `compiler` decorator with verbose flag.
    """

    configuration = helpers.configuration()
    artifacts = cnp.DebugArtifacts()

    @cnp.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    function.compile(inputset, configuration, artifacts, verbose=True, virtual=True)

    captured = capsys.readouterr()
    assert captured.out.strip() == (
        f"""

Computation Graph
------------------------------------------------------------------
{list(artifacts.textual_representations_of_graphs.values())[-1][-1]}
------------------------------------------------------------------

MLIR
------------------------------------------------------------------
Virtual circuits don't have MLIR.
------------------------------------------------------------------

Optimizer
------------------------------------------------------------------
Virtual circuits don't have optimizer output.
------------------------------------------------------------------

        """.strip()
    )


def test_circuit(helpers):
    """
    Test circuit decorator.
    """

    @cnp.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit1(x: cnp.uint2):
        return x + 42

    helpers.check_str(
        """

%0 = x                  # EncryptedScalar<uint2>
%1 = 42                 # ClearScalar<uint6>
%2 = add(%0, %1)        # EncryptedScalar<uint6>
return %2

        """.strip(),
        str(circuit1),
    )

    # ======================================================================

    @cnp.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit2(x: cnp.tensor[cnp.uint2, 3, 2]):
        return x + 42

    helpers.check_str(
        """

%0 = x                  # EncryptedTensor<uint2, shape=(3, 2)>
%1 = 42                 # ClearScalar<uint6>
%2 = add(%0, %1)        # EncryptedTensor<uint6, shape=(3, 2)>
return %2

        """.strip(),
        str(circuit2),
    )

    # ======================================================================

    @cnp.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit3(x: cnp.uint3):
        def square(x):
            return x**2

        return cnp.univariate(square, outputs=cnp.uint7)(x)

    helpers.check_str(
        """

%0 = x                 # EncryptedScalar<uint3>
%1 = square(%0)        # EncryptedScalar<uint7>
return %1

        """.strip(),
        str(circuit3),
    )

    # ======================================================================

    @cnp.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit4(x: cnp.uint3):
        return ((np.sin(x) ** 2) + (np.cos(x) ** 2)).astype(cnp.uint3)

    helpers.check_str(
        """

%0 = x                   # EncryptedScalar<uint3>
%1 = subgraph(%0)        # EncryptedScalar<uint3>
return %1

Subgraphs:

    %1 = subgraph(%0):

        %0 = input                # EncryptedScalar<uint3>
        %1 = sin(%0)              # EncryptedScalar<float64>
        %2 = 2                    # ClearScalar<uint2>
        %3 = power(%1, %2)        # EncryptedScalar<float64>
        %4 = cos(%0)              # EncryptedScalar<float64>
        %5 = 2                    # ClearScalar<uint2>
        %6 = power(%4, %5)        # EncryptedScalar<float64>
        %7 = add(%3, %6)          # EncryptedScalar<float64>
        %8 = astype(%7)           # EncryptedScalar<uint3>
        return %8

        """.strip(),
        str(circuit4),
    )

    # ======================================================================

    @cnp.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit5(x: cnp.int2):
        return x + 42

    helpers.check_str(
        """

%0 = x                  # EncryptedScalar<int2>
%1 = 42                 # ClearScalar<uint6>
%2 = add(%0, %1)        # EncryptedScalar<int6>
return %2

        """.strip(),
        str(circuit5),
    )


def test_bad_circuit(helpers):
    """
    Test circuit decorator with bad parameters.
    """

    # bad annotation
    # --------------

    with pytest.raises(ValueError) as excinfo:

        @cnp.circuit({"x": "encrypted"}, helpers.configuration())
        def circuit1(x: int):
            return x + 42

    assert str(excinfo.value) == (
        f"Annotation {str(int)} for argument 'x' is not valid "
        f"(please use a cnp type such as `cnp.uint4` or 'cnp.tensor[cnp.uint4, 3, 2]')"
    )

    # missing encryption status
    # -------------------------

    with pytest.raises(ValueError) as excinfo:

        @cnp.circuit({}, helpers.configuration())
        def circuit2(x: cnp.uint3):
            return x + 42

    assert str(excinfo.value) == (
        "Encryption status of parameter 'x' of function 'circuit2' is not provided"
    )

    # bad astype
    # ----------
    with pytest.raises(ValueError) as excinfo:

        @cnp.circuit({"x": "encrypted"}, helpers.configuration())
        def circuit3(x: cnp.uint3):
            return x.astype(np.int64)

    assert str(excinfo.value) == (
        "`astype` method must be called with a concrete.numpy type "
        "for direct circuit definition (e.g., value.astype(cnp.uint4))"
    )

    # round
    # -----
    with pytest.raises(RuntimeError) as excinfo:

        @cnp.circuit({"x": "encrypted"}, helpers.configuration())
        def circuit4(x: cnp.uint3):
            return round(x)

    assert str(excinfo.value) == (
        "'round(x)' cannot be used in direct definition (you may use np.around instead)"
    )
