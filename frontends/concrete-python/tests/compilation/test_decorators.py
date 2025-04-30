"""
Tests of `compiler` and `circuit` decorators.
"""

import numpy as np
import pytest

from concrete import fhe


def test_compiler_call_and_compile(helpers):
    """
    Test `__call__` and `compile` methods of `compiler` decorator back to back.
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    for i in range(10):
        function(i)

    circuit = function.compile(configuration=configuration)

    sample = 5
    helpers.check_execution(circuit, function, sample)


def test_compiler_verbose_compile(helpers, capsys, monkeypatch):
    """
    Test `compile` method of `compiler` decorator with verbose flag.
    """

    monkeypatch.setattr("concrete.fhe.compilation.artifacts.get_terminal_size", lambda: 80)

    configuration = helpers.configuration()
    artifacts = fhe.DebugArtifacts()

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    circuit = function.compile(inputset, configuration, artifacts, verbose=True)

    captured = capsys.readouterr()

    assert captured.out.strip().startswith(
        f"""


Computation Graph for function
--------------------------------------------------------------------------------
{circuit.graph.format()}
--------------------------------------------------------------------------------

MLIR
--------------------------------------------------------------------------------
{artifacts.mlir_to_compile}
--------------------------------------------------------------------------------

Bit-Width Constraints for function
--------------------------------------------------------------------------------
%0:
    function.%0 >= 4
%1:
    function.%1 >= 6
%2:
    function.%2 >= 6
    function.%0 == function.%1
    function.%1 == function.%2
--------------------------------------------------------------------------------

Bit-Width Assignments for function
--------------------------------------------------------------------------------
 function.%0 = 6
 function.%1 = 6
 function.%2 = 6
function.max = 6
--------------------------------------------------------------------------------

Bit-Width Assigned Computation Graph for function
--------------------------------------------------------------------------------
{circuit.graph.format(show_assigned_bit_widths=True)}
--------------------------------------------------------------------------------

Optimizer
--------------------------------------------------------------------------------

        """.strip()
    ), captured.out.strip()


def test_circuit(helpers):
    """
    Test circuit decorator.
    """

    @fhe.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit1(x: fhe.uint2):
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

    @fhe.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit2(x: fhe.tensor[fhe.uint2, 3, 2]):
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

    @fhe.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit3(x: fhe.uint3):
        def square(x):
            return x**2

        return fhe.univariate(square, outputs=fhe.uint7)(x)

    helpers.check_str(
        """

%0 = x                 # EncryptedScalar<uint3>
%1 = square(%0)        # EncryptedScalar<uint7>
return %1

        """.strip(),
        str(circuit3),
    )

    # ======================================================================

    @fhe.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit4(x: fhe.uint3):
        return ((np.sin(x) ** 2) + (np.cos(x) ** 2)).round().astype(fhe.uint3)

    helpers.check_str(
        """

%0 = x                   # EncryptedScalar<uint3>
%1 = subgraph(%0)        # EncryptedScalar<uint3>
return %1

Subgraphs:

    %1 = subgraph(%0):

        %0 = input                         # EncryptedScalar<uint3>
        %1 = sin(%0)                       # EncryptedScalar<float64>
        %2 = 2                             # ClearScalar<uint2>
        %3 = power(%1, %2)                 # EncryptedScalar<float64>
        %4 = cos(%0)                       # EncryptedScalar<float64>
        %5 = 2                             # ClearScalar<uint2>
        %6 = power(%4, %5)                 # EncryptedScalar<float64>
        %7 = add(%3, %6)                   # EncryptedScalar<float64>
        %8 = around(%7, decimals=0)        # EncryptedScalar<float64>
        %9 = astype(%8)                    # EncryptedScalar<uint3>
        return %9

        """.strip(),
        str(circuit4),
    )

    # ======================================================================

    @fhe.circuit({"x": "encrypted"}, helpers.configuration())
    def circuit5(x: fhe.int2):
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

    # ======================================================================

    @fhe.circuit({"x": "encrypted", "y": "encrypted"}, helpers.configuration())
    def circuit6(x: fhe.uint3, y: fhe.uint1):
        def keep(value, flag):
            return value if flag else 0

        return fhe.multivariate(keep, outputs=fhe.uint3)(x, y)

    helpers.check_str(
        """

%0 = x                   # EncryptedScalar<uint3>
%1 = y                   # EncryptedScalar<uint1>
%2 = keep(%0, %1)        # EncryptedScalar<uint3>
return %2

        """.strip(),
        str(circuit6),
    )


def test_bad_circuit(helpers):
    """
    Test circuit decorator with bad parameters.
    """

    # bad annotation
    # --------------

    with pytest.raises(ValueError) as excinfo:

        @fhe.circuit({"x": "encrypted"}, helpers.configuration())
        def circuit1(x: int):
            return x + 42

    assert str(excinfo.value) == (
        f"Annotation {str(int)} for argument 'x' is not valid "
        f"(please use an fhe type such as `fhe.uint4` or 'fhe.tensor[fhe.uint4, 3, 2]')"
    )

    # missing encryption status
    # -------------------------

    with pytest.raises(ValueError) as excinfo:

        @fhe.circuit({}, helpers.configuration())
        def circuit2(x: fhe.uint3):
            return x + 42

    assert str(excinfo.value) == (
        "Encryption status of parameter 'x' of function 'circuit2' is not provided"
    )

    # bad astype
    # ----------
    with pytest.raises(ValueError) as excinfo:

        @fhe.circuit({"x": "encrypted"}, helpers.configuration())
        def circuit3(x: fhe.uint3):
            return x.astype(np.int64)

    assert str(excinfo.value) == (
        "`astype` method must be called with an fhe type "
        "for direct circuit definition (e.g., value.astype(fhe.uint4))"
    )

    # round
    # -----
    with pytest.raises(RuntimeError) as excinfo:

        @fhe.circuit({"x": "encrypted"}, helpers.configuration())
        def circuit4(x: fhe.uint3):
            return round(x)

    assert str(excinfo.value) == (
        "'round(x)' cannot be used in direct definition (you may use np.around instead)"
    )


def test_compiler_reset(helpers):
    """
    Test compiler reset.
    """

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def compiler(x, y):
        return x + y

    configuration = helpers.configuration()

    inputset1 = fhe.inputset(fhe.uint3, fhe.uint3)
    circuit1 = compiler.compile(inputset1, configuration)

    helpers.check_str(
        """

module {
  func.func @compiler(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
    %0 = "FHE.add_eint"(%arg0, %arg1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    return %0 : !FHE.eint<4>
  }
}

        """.strip(),
        circuit1.mlir.strip(),
    )
    compiler.reset()

    inputset2 = fhe.inputset(fhe.uint10, fhe.uint10)
    circuit2 = compiler.compile(inputset2, configuration)

    helpers.check_str(
        """

module {
  func.func @compiler(%arg0: !FHE.eint<11>, %arg1: !FHE.eint<11>) -> !FHE.eint<11> {
    %0 = "FHE.add_eint"(%arg0, %arg1) : (!FHE.eint<11>, !FHE.eint<11>) -> !FHE.eint<11>
    return %0 : !FHE.eint<11>
  }
}

        """.strip(),
        circuit2.mlir.strip(),
    )
    compiler.reset()

    inputset3 = fhe.inputset(fhe.tensor[fhe.uint2, 3, 2], fhe.tensor[fhe.uint2, 2])  # type: ignore
    circuit3 = compiler.compile(inputset3, configuration)

    helpers.check_str(
        """

module {
  func.func @compiler(%arg0: tensor<3x2x!FHE.eint<3>>, %arg1: tensor<2x!FHE.eint<3>>) -> tensor<3x2x!FHE.eint<3>> {
    %0 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x2x!FHE.eint<3>>, tensor<2x!FHE.eint<3>>) -> tensor<3x2x!FHE.eint<3>>
    return %0 : tensor<3x2x!FHE.eint<3>>
  }
}

        """.strip(),  # noqa: E501
        circuit3.mlir.strip(),
    )
    compiler.reset()
