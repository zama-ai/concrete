"""
Tests of `Compiler` class.
"""

import json

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.compilation import ClientSpecs, Compiler


def test_compiler_bad_init():
    """
    Test `__init__` method of `Compiler` class with bad parameters.
    """

    def f(x, y, z):
        return x + y + z

    # missing all
    # -----------

    with pytest.raises(ValueError) as excinfo:
        Compiler(f, {})

    assert str(excinfo.value) == (
        "Encryption statuses of parameters 'x', 'y' and 'z' of function 'f' are not provided"
    )

    # missing x and y
    # ---------------

    with pytest.raises(ValueError) as excinfo:
        Compiler(f, {"z": "clear"})

    assert str(excinfo.value) == (
        "Encryption statuses of parameters 'x' and 'y' of function 'f' are not provided"
    )

    # missing x
    # ---------

    with pytest.raises(ValueError) as excinfo:
        Compiler(f, {"y": "encrypted", "z": "clear"})

    assert str(excinfo.value) == (
        "Encryption status of parameter 'x' of function 'f' is not provided"
    )

    # additional a, b, c
    # ------------------
    with pytest.raises(ValueError) as excinfo:
        Compiler(
            f,
            {
                "x": "encrypted",
                "y": "encrypted",
                "z": "encrypted",
                "a": "encrypted",
                "b": "encrypted",
                "c": "encrypted",
            },
        )

    assert str(excinfo.value) == (
        "Encryption statuses of 'a', 'b' and 'c' are provided "
        "but they are not a parameter of function 'f'"
    )

    # additional a and b
    # ------------------

    with pytest.raises(ValueError) as excinfo:
        Compiler(
            f,
            {
                "x": "encrypted",
                "y": "encrypted",
                "z": "encrypted",
                "a": "encrypted",
                "b": "encrypted",
            },
        )

    assert str(excinfo.value) == (
        "Encryption statuses of 'a' and 'b' are provided "
        "but they are not a parameter of function 'f'"
    )

    # additional a
    # ------------

    with pytest.raises(ValueError) as excinfo:
        Compiler(
            f,
            {
                "x": "encrypted",
                "y": "encrypted",
                "z": "encrypted",
                "a": "encrypted",
            },
        )

    assert str(excinfo.value) == (
        "Encryption status of 'a' is provided but it is not a parameter of function 'f'"
    )


def test_compiler_bad_call():
    """
    Test `__call__` method of `Compiler` class with bad parameters.
    """

    def f(x, y, z):
        return x + y + z

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(f, {"x": "encrypted", "y": "encrypted", "z": "clear"})
        compiler(1, 2, 3, invalid=4)

    assert str(excinfo.value) == "Calling function 'f' with kwargs is not supported"


def test_compiler_bad_trace(helpers):
    """
    Test `trace` method of `Compiler` class with bad parameters.
    """

    configuration = helpers.configuration()

    # without inputset
    # ----------------

    def f(x, y, z):
        return x + y + z

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
        )
        compiler.trace(configuration=configuration)

    assert str(excinfo.value) == "Tracing function 'f' without an inputset is not supported"

    # bad return
    # ----------

    def g():
        return np.array([{}, ()], dtype=object)

    with pytest.raises(ValueError) as excinfo:
        compiler = Compiler(g, {})
        compiler.trace(inputset=[()], configuration=configuration)

    assert str(excinfo.value) == "Function 'g' returned '[{} ()]', which is not supported"

    # len on scalar
    # -------------

    def len_on_scalar(x):
        return len(x)

    with pytest.raises(TypeError) as excinfo:
        compiler = Compiler(len_on_scalar, {"x": "encrypted"})
        compiler.trace(inputset=[1, 2, 3], configuration=configuration)

    assert str(excinfo.value) == "object of type 'Tracer' where 'shape == ()' has no len()"


def test_compiler_bad_compile(helpers):
    """
    Test `compile` method of `Compiler` class with bad parameters.
    """

    configuration = helpers.configuration()

    def f(x, y, z):
        return x + y + z

    # without inputset
    # ----------------

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
        )
        compiler.compile(configuration=configuration)

    assert str(excinfo.value) == "Compiling function 'f' without an inputset is not supported"

    # with bad inputset at the first input
    # ------------------------------------

    with pytest.raises(ValueError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
        )
        inputset = [1]
        compiler.compile(inputset, configuration=configuration)

    assert str(excinfo.value) == (
        "Input #0 of your inputset is not well formed "
        "(expected a tuple of 3 values got a single value)"
    )

    # with bad inputset at the second input
    # -------------------------------------

    with pytest.raises(ValueError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
        )
        inputset = [(1, 2, 3), (1, 2)]
        compiler.compile(inputset, configuration=configuration)

    assert str(excinfo.value) == (
        "Input #1 of your inputset is not well formed "
        "(expected a tuple of 3 values got a tuple of 2 values)"
    )


def test_compiler_compile_bad_inputset(helpers):
    """
    Test `compile` method of `Compiler` class with bad inputset.
    """

    configuration = helpers.configuration()

    # with inf
    # --------

    def f(x):
        return (x + np.inf).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(f, {"x": "encrypted"})
        compiler.compile(range(10), configuration=configuration)

    assert str(excinfo.value) == "Bound measurement using inputset[0] failed"

    helpers.check_str(
        """

Evaluation of the graph failed

%0 = x                   # EncryptedScalar<uint1>
%1 = subgraph(%0)        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ evaluation of this node failed
return %1

Subgraphs:

    %1 = subgraph(%0):

        %0 = input                         # EncryptedScalar<uint1>
        %1 = inf                           # ClearScalar<float64>
        %2 = add(%0, %1)                   # EncryptedScalar<float64>
        %3 = astype(%2, dtype=int_)        # EncryptedScalar<uint1>
        return %3

    """.strip(),
        str(excinfo.value.__cause__).strip(),
    )

    helpers.check_str(
        """

Evaluation of the graph failed

%0 = input                         # EncryptedScalar<uint1>
%1 = inf                           # ClearScalar<float64>
%2 = add(%0, %1)                   # EncryptedScalar<float64>
%3 = astype(%2, dtype=int_)        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ evaluation of this node failed
return %3

    """.strip(),
        str(excinfo.value.__cause__.__cause__).strip(),
    )

    assert (
        str(excinfo.value.__cause__.__cause__.__cause__)
        == "An `Inf` value is tried to be converted to integer"
    )

    # with nan
    # --------

    def g(x):
        return (x + np.nan).astype(np.int64)

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(g, {"x": "encrypted"})
        compiler.compile(range(10), configuration=configuration)

    assert str(excinfo.value) == "Bound measurement using inputset[0] failed"

    helpers.check_str(
        """

Evaluation of the graph failed

%0 = x                   # EncryptedScalar<uint1>
%1 = subgraph(%0)        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ evaluation of this node failed
return %1

Subgraphs:

    %1 = subgraph(%0):

        %0 = input                         # EncryptedScalar<uint1>
        %1 = nan                           # ClearScalar<float64>
        %2 = add(%0, %1)                   # EncryptedScalar<float64>
        %3 = astype(%2, dtype=int_)        # EncryptedScalar<uint1>
        return %3

    """.strip(),
        str(excinfo.value.__cause__).strip(),
    )

    helpers.check_str(
        """

Evaluation of the graph failed

%0 = input                         # EncryptedScalar<uint1>
%1 = nan                           # ClearScalar<float64>
%2 = add(%0, %1)                   # EncryptedScalar<float64>
%3 = astype(%2, dtype=int_)        # EncryptedScalar<uint1>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ evaluation of this node failed
return %3

    """.strip(),
        str(excinfo.value.__cause__.__cause__).strip(),
    )

    assert (
        str(excinfo.value.__cause__.__cause__.__cause__)
        == "A `NaN` value is tried to be converted to integer"
    )


def test_compiler_compile_with_single_tuple_inputset(helpers):
    """
    Test compiling a single argument function with an inputset made of single element tuples.
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x

    inputset = [(3,), (4,), (5,)]
    circuit = f.compile(inputset, configuration)

    sample = 4
    helpers.check_execution(circuit, f, sample)


def test_compiler_tampered_client_parameters(helpers):
    """
    Test running a function with tampered client parameters.
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x

    inputset = [(3,), (4,), (5,)]
    circuit = f.compile(inputset, configuration)
    sample = 4

    client_parameters_json = json.loads(circuit.client.specs.serialize())
    client_parameters_json["circuits"][0]["inputs"][0]["typeInfo"] = {}

    tampered_bytes = bytes(json.dumps(client_parameters_json), "UTF-8")
    circuit.client.specs = ClientSpecs.deserialize(tampered_bytes)

    with pytest.raises(ValueError) as excinfo:
        helpers.check_execution(circuit, f, sample)
    assert str(excinfo.value) == "Expected a valid type in dict_keys([])"


def test_compiler_enable_fusing(helpers):
    """
    Test compilation with and without TLU fusing.
    """

    # Make sure it's enabled by default
    default_configuration = fhe.Configuration()
    assert default_configuration.enable_tlu_fusing

    def f(x):
        return (x**2) // 2

    # Fused Scalar
    compiler1 = Compiler(f, {"x": "encrypted"})
    circuit1 = compiler1.compile(
        fhe.inputset(fhe.uint3),
        helpers.configuration().fork(enable_tlu_fusing=True),
    )
    assert circuit1.programmable_bootstrap_count == 1

    # Not Fused Scalar
    compiler2 = Compiler(f, {"x": "encrypted"})
    circuit2 = compiler2.compile(
        fhe.inputset(fhe.uint3),
        helpers.configuration().fork(enable_tlu_fusing=False),
    )
    assert circuit2.programmable_bootstrap_count == 2

    # Fused Tensor
    compiler3 = Compiler(f, {"x": "encrypted"})
    circuit3 = compiler3.compile(
        fhe.inputset(fhe.tensor[fhe.uint3, 3]),  # type: ignore
        helpers.configuration().fork(enable_tlu_fusing=True),
    )
    assert circuit3.programmable_bootstrap_count == 3

    # Not Fused Tensor
    compiler4 = Compiler(f, {"x": "encrypted"})
    circuit4 = compiler4.compile(
        fhe.inputset(fhe.tensor[fhe.uint3, 3]),  # type: ignore
        helpers.configuration().fork(enable_tlu_fusing=False),
    )
    assert circuit4.programmable_bootstrap_count == 6


def test_compiler_reset(helpers):
    """
    Test compiler reset.
    """

    def f(x, y):
        return x + y

    configuration = helpers.configuration()
    compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})

    inputset1 = fhe.inputset(fhe.uint3, fhe.uint3)
    circuit1 = compiler.compile(inputset1, configuration)

    helpers.check_str(
        """

module {
  func.func @f(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
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
  func.func @f(%arg0: !FHE.eint<11>, %arg1: !FHE.eint<11>) -> !FHE.eint<11> {
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
  func.func @f(%arg0: tensor<3x2x!FHE.eint<3>>, %arg1: tensor<2x!FHE.eint<3>>) -> tensor<3x2x!FHE.eint<3>> {
    %0 = "FHELinalg.add_eint"(%arg0, %arg1) : (tensor<3x2x!FHE.eint<3>>, tensor<2x!FHE.eint<3>>) -> tensor<3x2x!FHE.eint<3>>
    return %0 : tensor<3x2x!FHE.eint<3>>
  }
}

        """.strip(),  # noqa: E501
        circuit3.mlir.strip(),
    )
    compiler.reset()
