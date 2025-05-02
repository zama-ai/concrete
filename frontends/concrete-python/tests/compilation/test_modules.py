"""
Tests of everything related to modules.
"""

import inspect
import tempfile
from concurrent.futures import Future
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from concrete import fhe

# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-argument, unused-variable, no-member, unused-argument, function-redefined, expression-not-assigned
# same disables for ruff:
# ruff: noqa: N805, E501, F841, ARG002, F811, B015


def test_get_wrong_attribute():
    """
    Test that getting the wrong attribute fails.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def square(x):
            return x**2

    with pytest.raises(AttributeError, match="No attribute nothing"):
        a = Module.nothing


def test_empty_module():
    """
    Test that defining a module without functions is an error.
    """
    with pytest.raises(
        RuntimeError, match="Tried to define an @fhe.module without any @fhe.function"
    ):

        @fhe.module()
        class Module:
            def square(x):
                return x**2


def test_wrong_info():
    """
    Test that defining a module with wrong information raises an error.
    """

    with pytest.raises(ValueError) as excinfo:

        @fhe.module()
        class Module:
            @fhe.function({"x": "encrypted"})
            def add(x, y, z, w):
                return x + y

    assert str(excinfo.value) == (
        "Encryption statuses of parameters 'y', 'z' and 'w' of function 'add' are not provided"
    )

    with pytest.raises(ValueError) as excinfo:

        @fhe.module()
        class Module:
            @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted", "w": "encrypted"})
            def add(x):
                return x

    assert str(excinfo.value) == (
        "Encryption statuses of 'y', 'z' and 'w' are provided but they are not a parameter of function 'add'"
    )


def test_wrong_inputset(helpers):
    """
    Test that a wrong inputset raises an error.
    """

    with pytest.raises(ValueError) as excinfo:

        @fhe.module()
        class Module:
            @fhe.function({"x": "encrypted", "y": "encrypted"})
            def add(x, y):
                return x + y

        inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
        module = Module.compile(
            {"add": inputset},
        )

    assert str(excinfo.value) == (
        "Input #0 of your inputset is not well formed (expected a tuple of 2 values got a single value)"
    )

    with pytest.raises(RuntimeError) as excinfo:

        @fhe.module()
        class Module:
            @fhe.function({"x": "encrypted", "y": "encrypted"})
            def add(x, y):
                return x + y

        inputset = []
        module = Module.compile(
            {"add": inputset},
        )

    assert str(excinfo.value) == ("Compiling function 'add' without an inputset is not supported")


def test_non_composable_message():
    """
    Test the non composable Dag message.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def add(x, y):
            return x + y

    line = inspect.currentframe().f_lineno - 2
    with pytest.raises(RuntimeError) as excinfo:
        Module.compile({"add": [(0, 0), (3, 3)]})

    assert (
        str(excinfo.value)
        == f"""\
Program can not be composed (see https://docs.zama.ai/concrete/compilation/common_errors#id-9.-non-composable-circuit): \
At location test_modules.py:{line}:0:\nThe noise of the node 0 is contaminated by noise coming straight from the input \
(partition: 0, coeff: 2.00).\
"""
    )


def test_call_clear_circuits():
    """
    Test that calling clear functions works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def square(x):
            return x**2

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def add_sub(x, y):
            return (x + y), (x - y)

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def mul(x, y):
            return x * y

    assert Module.square(2) == 4
    assert Module.add_sub(2, 3) == (5, -1)
    assert Module.mul(3, 4) == 12


def test_call_clear_circuits_wrong_kwargs():
    """
    Test that calling clear functions works.
    """

    with pytest.raises(RuntimeError) as excinfo:

        @fhe.module()
        class Module:
            @fhe.function({"x": "encrypted"})
            def square(x):
                return x**2

        Module.square(x=2) == 4
    assert str(excinfo.value) == ("Calling function 'square' with kwargs is not supported")


def test_autorounder(helpers):
    rounder1 = fhe.AutoRounder(target_msbs=5)

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def function1(x):
            y = x + 1000
            z = fhe.round_bit_pattern(y, lsbs_to_remove=rounder1)
            return np.sqrt(z).astype(np.int64)

    inputset1 = range(1000)
    fhe.AutoRounder.adjust(Module.function1, inputset1)
    assert rounder1.lsbs_to_remove == 6

    module = Module.compile({"function1": inputset1}, auto_adjust_rounders=True)


def test_autotruncator(helpers):
    truncator1 = fhe.AutoTruncator(target_msbs=5)

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def function1(x):
            y = x + 1000
            z = fhe.truncate_bit_pattern(y, lsbs_to_remove=truncator1)
            return np.sqrt(z).astype(np.int64)

    inputset1 = range(1000)
    fhe.AutoTruncator.adjust(Module.function1, inputset1)
    assert truncator1.lsbs_to_remove == 6

    module = Module.compile(
        {"function1": inputset1},
        auto_adjust_truncators=True,
    )


def test_compile():
    """
    Test that compiling a module works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    Module.compile({"inc": inputset, "dec": inputset}, verbose=True)
    artifacts = fhe.ModuleDebugArtifacts()
    with tempfile.TemporaryDirectory() as tmp:
        module = Module.compile(
            {"inc": inputset, "dec": inputset},
            module_artifacts=artifacts,
            verbose=True,
            use_insecure_key_cache=True,
            enable_unsafe_features=True,
            insecure_key_cache_location=tmp,
        )

        assert module.mlir is not None
        assert module.keys is not None
        module.keygen()
        module.cleanup()
        assert set(artifacts.functions.keys()) == {"inc", "dec"}

        assert repr(module.inc) == "FheFunction(name=inc)"
        assert repr(module.dec) == "FheFunction(name=dec)"


def test_compiled_wrong_attribute():
    """
    Test that getting unexisting attribute on module fails
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    Module.compile({"inc": inputset, "dec": inputset}, verbose=True)
    artifacts = fhe.ModuleDebugArtifacts()

    module = Module.compile(
        {"inc": inputset, "dec": inputset},
        module_artifacts=artifacts,
        verbose=True,
    )

    with pytest.raises(AttributeError, match="No attribute nothing"):
        a = module.nothing


def test_compiled_clear_call():
    """
    Test that cleartext execution works on compiled objects.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile(
        {"inc": inputset, "dec": inputset},
    )

    assert module.inc(5) == 6
    assert module.dec(5) == 4


def test_compiled_simulation(helpers):
    """
    Test that simulation works on compiled objects.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]

    module = Module.compile(
        {"inc": inputset, "dec": inputset},
        fhe_simulation=True,
    )

    assert module.inc.simulate(5) == 6
    assert module.dec.simulate(5) == 4

    module.cleanup()


@pytest.mark.graphviz
def test_print(helpers):
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    inputset = list(range(20))
    module = Module.compile(
        {"inc": inputset},
    )
    helpers.check_str(
        """
%0 = x                        # EncryptedScalar<uint5>        ∈ [0, 19]
%1 = 1                        # ClearScalar<uint1>            ∈ [1, 1]
%2 = add(%0, %1)              # EncryptedScalar<uint5>        ∈ [1, 20]
%3 = 20                       # ClearScalar<uint5>            ∈ [20, 20]
%4 = remainder(%2, %3)        # EncryptedScalar<uint5>        ∈ [0, 19]
return %4
        """,
        str(module.inc),
    )


def test_encrypted_execution():
    """
    Test that encrypted execution works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return (x - 1) % 20

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile(
        {"inc": inputset, "dec": inputset},
    )

    x = 5
    x_enc = module.inc.encrypt(x)
    x_inc_enc = module.inc.run(x_enc)
    x_inc = module.inc.decrypt(x_inc_enc)
    assert x_inc == 6

    assert module.inc.encrypt_run_decrypt(2) == 3

    x_inc_dec_enc = module.dec.run(x_inc_enc)
    x_inc_dec = module.dec.decrypt(x_inc_dec_enc)
    assert x_inc_dec == 5

    for _ in range(10):
        x_enc = module.inc.run(x_enc)
    x_dec = module.inc.decrypt(x_enc)
    assert x_dec == 15


def test_key_set():
    """
    Test that keys can be set.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return (x - 1) % 20

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile(
        {"inc": inputset, "dec": inputset},
    )

    keys = module.keys
    module.keygen(force=True)
    module.keys = keys

    x = 5
    x_enc = module.inc.encrypt(x)
    x_inc_enc = module.inc.run(x_enc)
    x_inc = module.inc.decrypt(x_inc_enc)
    assert x_inc == 6

    assert module.inc.encrypt_run_decrypt(2) == 3

    x_inc_dec_enc = module.dec.run(x_inc_enc)
    x_inc_dec = module.dec.decrypt(x_inc_dec_enc)
    assert x_inc_dec == 5

    for _ in range(10):
        x_enc = module.inc.run(x_enc)
    x_dec = module.inc.decrypt(x_enc)
    assert x_dec == 15


def test_composition_policy_default():
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def square(x):
            return x**2

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def add_sub(x, y):
            return (x + y), (x - y)

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def mul(x, y):
            return x * y

    assert isinstance(Module.composition, fhe.CompositionPolicy)
    assert isinstance(Module.composition, fhe.AllComposable)


def test_composition_policy_all_composable():
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def square(x):
            return x**2

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def add_sub(x, y):
            return (x + y), (x - y)

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def mul(x, y):
            return x * y

        composition = fhe.AllComposable()

    assert isinstance(Module.composition, fhe.CompositionPolicy)
    assert isinstance(Module.composition, fhe.AllComposable)


def test_composition_policy_wires():
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def square(x):
            return x**2

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def add_sub(x, y):
            return (x + y), (x - y)

        composition = fhe.Wired(
            {
                fhe.Wire(fhe.AllOutputs(add_sub), fhe.AllInputs(add_sub)),
                fhe.Wire(fhe.AllOutputs(add_sub), fhe.Input(square, 0)),
            }
        )

    assert isinstance(Module.composition, fhe.CompositionPolicy)
    assert isinstance(Module.composition, fhe.Wired)


def test_composition_wired_enhances_complexity():
    @fhe.module()
    class Module1:
        @fhe.function({"x": "encrypted"})
        def _1(x):
            return (x * 2) % 20

        @fhe.function({"x": "encrypted"})
        def _2(x):
            return (x * 2) % 200

        composition = fhe.Wired(
            {
                fhe.Wire(fhe.Output(_1, 0), fhe.Input(_2, 0)),
            }
        )

    module1 = Module1.compile(
        {
            "_1": [np.random.randint(1, 20, size=()) for _ in range(100)],
            "_2": [np.random.randint(1, 200, size=()) for _ in range(100)],
        },
    )

    @fhe.module()
    class Module2:
        @fhe.function({"x": "encrypted"})
        def _1(x):
            return (x * 2) % 20

        @fhe.function({"x": "encrypted"})
        def _2(x):
            return (x * 2) % 200

        composition = fhe.AllComposable()

    module2 = Module2.compile(
        {
            "_1": [np.random.randint(1, 20, size=()) for _ in range(100)],
            "_2": [np.random.randint(1, 200, size=()) for _ in range(100)],
        },
    )

    assert module1.complexity < module2.complexity


def test_composition_wired_compilation():
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def a(x):
            return (x * 2) % 20

        @fhe.function({"x": "encrypted"})
        def b(x):
            return (x * 2) % 50

        @fhe.function({"x": "encrypted"})
        def c(x):
            return (x * 2) % 100

        composition = fhe.Wired(
            {
                fhe.Wire(fhe.Output(a, 0), fhe.Input(b, 0)),
                fhe.Wire(fhe.Output(b, 0), fhe.Input(c, 0)),
            }
        )

    module = Module.compile(
        {
            "a": [np.random.randint(1, 20, size=()) for _ in range(100)],
            "b": [np.random.randint(1, 50, size=()) for _ in range(100)],
            "c": [np.random.randint(1, 100, size=()) for _ in range(100)],
        },
    )

    inp_enc = module.a.encrypt(5)
    a_enc = module.a.run(inp_enc)
    assert module.a.decrypt(a_enc) == 10
    b_enc = module.b.run(a_enc)
    assert module.b.decrypt(b_enc) == 20
    c_enc = module.c.run(b_enc)
    assert module.c.decrypt(c_enc) == 40


def test_simulate_encrypt_run_decrypt(helpers):
    """
    Test `simulate_encrypt_run_decrypt` configuration option.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return (x - 1) % 20

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile(
        {"inc": inputset, "dec": inputset},
        helpers.configuration().fork(
            fhe_execution=False,
            fhe_simulation=True,
            simulate_encrypt_run_decrypt=True,
        ),
    )

    sample_x = 10
    encrypted_x = module.inc.encrypt(sample_x)

    encrypted_result = module.inc.run(encrypted_x)
    result = module.inc.decrypt(encrypted_result)
    assert result == 11

    # Make sure computation happened in simulation.
    assert isinstance(encrypted_x, int)
    assert module.inc.simulation_runtime.initialized
    assert isinstance(encrypted_result, int)

    encrypted_result = module.dec.run(encrypted_result)
    result = module.dec.decrypt(encrypted_result)
    assert result == 10

    # Make sure computation happened in simulation.
    assert module.dec.simulation_runtime.initialized
    assert isinstance(encrypted_result, int)


def test_non_composable_due_to_increasing_noise():
    """
    Test that non composable module can be fixed with `fhe.refresh`.
    """

    inputsets = {"a": [(x, y) for x in range(16) for y in range(16)]}

    @fhe.module()
    class Broken:
        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def a(x, y):
            return fhe.identity(x + y)  # identity is optimized out in that case

    with pytest.raises(RuntimeError, match="Program can not be composed"):
        Broken.compile(inputsets)

    @fhe.module()
    class Fixed:
        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def a(x, y):
            return fhe.refresh(x + y)

    assert Fixed.compile(inputsets)


@pytest.mark.minimal
def test_client_server_api(helpers):
    """
    Test client/server API of modules.
    """

    configuration = helpers.configuration()

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile({"inc": inputset, "dec": inputset}, verbose=True)

    module.keygen()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        server_path = tmp_dir_path / "server.zip"
        module.server.save(server_path)

        client_path = tmp_dir_path / "client.zip"
        module.client.save(client_path)

        server = fhe.Server.load(server_path)

        serialized_client_specs = server.client_specs.serialize()
        client_specs = fhe.ClientSpecs.deserialize(serialized_client_specs)

        clients = [
            fhe.Client(client_specs, configuration.insecure_key_cache_location),
            fhe.Client.load(client_path, configuration.insecure_key_cache_location),
        ]

        for client in clients:
            arg = client.encrypt(10, function_name="inc")

            serialized_arg = arg.serialize()
            serialized_evaluation_keys = client.evaluation_keys.serialize()

            deserialized_arg = fhe.Value.deserialize(serialized_arg)
            deserialized_evaluation_keys = fhe.EvaluationKeys.deserialize(
                serialized_evaluation_keys,
            )

            result = server.run(
                deserialized_arg,
                evaluation_keys=deserialized_evaluation_keys,
                function_name="inc",
            )
            serialized_result = result.serialize()

            deserialized_result = fhe.Value.deserialize(serialized_result)
            output = client.decrypt(deserialized_result, function_name="inc")

            assert output == 11


def test_trace_wire_single_input_output(helpers):
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def a(x):
            return (x * 2) % 20

        @fhe.function({"x": "encrypted"})
        def b(x):
            return (x * 2) % 50

        @fhe.function({"x": "encrypted"})
        def c(x):
            return (x * 2) % 100

        composition = fhe.Wired()

    # `wire_pipeline` takes an inputset as input, activates a context in which wiring is recorded, and returns an iterator that can be used inside of the context to iterate over samples.
    with Module.wire_pipeline([np.random.randint(1, 20, size=()) for _ in range(100)]) as samples:
        for s in samples:
            Module.c(Module.b(Module.a(s)))

    assert len(Module.composition.wires) == 2
    assert fhe.Wire(fhe.Output(Module.a, 0), fhe.Input(Module.b, 0)) in Module.composition.wires
    assert fhe.Wire(fhe.Output(Module.b, 0), fhe.Input(Module.c, 0)) in Module.composition.wires

    module = Module.compile(
        p_error=0.000001,
    )

    inp_enc = module.a.encrypt(5)
    a_enc = module.a.run(inp_enc)
    assert module.a.decrypt(a_enc) == 10
    b_enc = module.b.run(a_enc)
    assert module.b.decrypt(b_enc) == 20
    c_enc = module.c.run(b_enc)
    assert module.c.decrypt(c_enc) == 40


def test_trace_wires_multi_inputs_outputs(helpers):
    @fhe.module()
    class Module:

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def a(x, y):
            return ((x + y) * 2) % 20, ((x - y) * 2) % 20

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def b(x, y):
            return ((x + y) * 2) % 20, ((x - y) * 2) % 20

        composition = fhe.Wired()

    # `wire_pipeline` takes an inputset as input, activates a context in which wiring is recorded, and returns an iterator that can be used inside of the context to iterate over samples.
    with Module.wire_pipeline(
        [(np.random.randint(1, 20, size=()), np.random.randint(1, 20, size=())) for _ in range(100)]
    ) as samples:
        for s in samples:
            output = Module.a(s[0], s[1])
            Module.b(*output)

    assert len(Module.composition.wires) == 2
    assert fhe.Wire(fhe.Output(Module.a, 0), fhe.Input(Module.b, 0)) in Module.composition.wires
    assert fhe.Wire(fhe.Output(Module.a, 1), fhe.Input(Module.b, 1)) in Module.composition.wires

    module = Module.compile(
        p_error=0.000001,
    )

    inp_enc = module.a.encrypt(5, 1)
    a_enc = module.a.run(*inp_enc)
    assert module.a.decrypt(a_enc) == (12, 8)
    b_enc = module.b.run(*a_enc)
    assert module.b.decrypt(b_enc) == (0, 8)


def test_lazy_simulation_execution(helpers):

    @fhe.module()
    class Module1:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    module = Module1.compile(
        {"inc": [np.random.randint(1, 20, size=()) for _ in range(100)]},
        helpers.configuration().fork(
            fhe_execution=False,
            fhe_simulation=False,
        ),
    )

    assert not module.execution_runtime.initialized
    assert not module.simulation_runtime.initialized
    result = module.inc.encrypt_run_decrypt(10)
    sim_res = module.inc.simulate(10)
    assert module.execution_runtime.initialized
    assert module.simulation_runtime.initialized

    @fhe.module()
    class Module2:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    module = Module2.compile(
        {"inc": [np.random.randint(1, 20, size=()) for _ in range(100)]},
        helpers.configuration().fork(
            fhe_execution=True,
            fhe_simulation=False,
        ),
    )

    assert module.execution_runtime.initialized
    assert not module.simulation_runtime.initialized
    sim_res = module.inc.simulate(10)
    assert module.simulation_runtime.initialized

    @fhe.module()
    class Module3:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    module = Module3.compile(
        {"inc": [np.random.randint(1, 20, size=()) for _ in range(100)]},
        helpers.configuration().fork(
            fhe_execution=False,
            fhe_simulation=True,
        ),
    )

    assert not module.execution_runtime.initialized
    assert module.simulation_runtime.initialized
    result = module.inc.encrypt_run_decrypt(10)
    assert module.execution_runtime.initialized

    @fhe.module()
    class Module4:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    module = Module4.compile(
        {"inc": [np.random.randint(1, 20, size=()) for _ in range(100)]},
        helpers.configuration().fork(
            fhe_execution=True,
            fhe_simulation=True,
        ),
    )

    assert module.execution_runtime.initialized
    assert module.simulation_runtime.initialized


def test_all_composable_with_clears(helpers):

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted", "y": "clear"})
        def inc(x, y):
            return (x + y + 1) % 20

    module = Module.compile(
        {
            "inc": [
                (np.random.randint(1, 20, size=()), np.random.randint(1, 20, size=()))
                for _ in range(100)
            ]
        },
        helpers.configuration().fork(),
    )


def test_wired_with_invalid_wire(helpers):

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted", "y": "clear"})
        def inc(x, y):
            return (x + y + 1) % 20

        composition = fhe.Wired(
            {
                fhe.Wire(fhe.Output(inc, 0), fhe.Input(inc, 0)),
                fhe.Wire(fhe.Output(inc, 0), fhe.Input(inc, 1)),  # Faulty one
            }
        )

    with pytest.raises(
        Exception, match="Invalid composition rule encountered: Input 0 of inc is not encrypted"
    ):
        module = Module.compile(
            {
                "inc": [
                    (np.random.randint(1, 20, size=()), np.random.randint(1, 20, size=()))
                    for _ in range(100)
                ]
            },
            helpers.configuration().fork(),
        )


def test_wired_with_all_encrypted_inputs(helpers):

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted", "y": "clear"})
        def inc(x, y):
            return (x + y + 1) % 20

        composition = fhe.Wired(
            {
                fhe.Wire(fhe.Output(inc, 0), fhe.AllInputs(inc)),
            }
        )

    module = Module.compile(
        {
            "inc": [
                (np.random.randint(1, 20, size=()), np.random.randint(1, 20, size=()))
                for _ in range(100)
            ]
        },
        helpers.configuration().fork(),
    )


class IncDec:
    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return fhe.refresh(x + 1)

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return fhe.refresh(x - 1)

    precision = 4

    inputset: ClassVar[list] = list(range(1, 2**precision - 1))
    to_compile: ClassVar[dict[str, list]] = {"inc": inputset, "dec": inputset}


def test_run_async():
    """
    Test `run_async` with `auto_schedule_run=False` configuration option.
    """

    module = IncDec.Module.compile(IncDec.to_compile)

    sample_x = 2
    encrypted_x = module.inc.encrypt(sample_x)

    a = module.inc.run_async(encrypted_x)
    assert isinstance(a, Future)

    b = module.dec.run(a)
    assert isinstance(b, type(encrypted_x))

    result = module.inc.decrypt(b)
    assert result == sample_x
    del module


def test_run_sync():
    """
    Test `run_sync` with `auto_schedule_run=True` configuration option.
    """

    conf = fhe.Configuration(auto_schedule_run=True)
    module = IncDec.Module.compile(IncDec.to_compile, conf)

    sample_x = 2
    encrypted_x = module.inc.encrypt(sample_x)

    a = module.inc.run(encrypted_x)
    assert isinstance(a, Future)

    b = module.dec.run_sync(a)
    assert isinstance(b, type(encrypted_x))

    result = module.inc.decrypt(b)
    assert result == sample_x


def test_run_auto_schedule():
    """
    Test `run` with `auto_schedule_run=True` configuration option.
    """

    conf = fhe.Configuration(auto_schedule_run=True)
    module = IncDec.Module.compile(IncDec.to_compile, conf)

    sample_x = 2
    encrypted_x = module.inc.encrypt(sample_x)

    a = module.inc.run(encrypted_x)
    assert isinstance(a, Future)

    b = module.dec.run(a)
    assert isinstance(b, Future)

    result = module.inc.decrypt(b)
    assert result == sample_x
