"""
Tests of `Compiler` class.
"""

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.compilation import Compiler


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


def test_direct_circuit_multi_output(helpers):
    """
    Test `run` method of `Circuit` class with bad parameters.
    """

    configuration = helpers.configuration()
    with pytest.raises(RuntimeError) as excinfo:

        @fhe.circuit({"x": "encrypted", "y": "encrypted"}, configuration)
        def f(x: fhe.uint4, y: fhe.uint4):
            return x, y

    assert (
        str(excinfo.value)
        == """\
Function you are trying to compile cannot be compiled

%0 = x        # EncryptedScalar<uint4>
%1 = y        # EncryptedScalar<uint4>
return %0, %1
^^^^^^^^^^^^^ multiple outputs are not supported\
"""
    )


def test_circuit_multi_output(helpers):
    """
    Test `run` method of `Circuit` class with bad parameters.
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x, y

    inputset = [(0, 0), (15, 15)]
    graph = f.trace(inputset)
    assert (
        graph.format()
        == """\
%0 = x        # EncryptedScalar<uint4>        ∈ [0, 15]
%1 = y        # EncryptedScalar<uint4>        ∈ [0, 15]
return %0, %1\
"""
    )

    with pytest.raises(RuntimeError) as excinfo:
        f.compile(inputset, configuration)
    assert (
        str(excinfo.value)
        == """\
Function you are trying to compile cannot be compiled

%0 = x        # EncryptedScalar<uint4>
%1 = y        # EncryptedScalar<uint4>
return %0, %1
^^^^^^^^^^^^^ multiple outputs are not supported\
"""
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
