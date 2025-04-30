"""
Tests of `Configuration` class.
"""

import os
import sys

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.compilation import Configuration
from concrete.fhe.compilation.configuration import SecurityLevel

from ..conftest import USE_MULTI_PRECISION


@pytest.mark.parametrize(
    "kwargs,expected_error,expected_message",
    [
        pytest.param(
            {"enable_unsafe_features": False, "use_insecure_key_cache": True},
            RuntimeError,
            "Insecure key cache cannot be used without enabling unsafe features",
        ),
        pytest.param(
            {
                "enable_unsafe_features": True,
                "use_insecure_key_cache": True,
                "insecure_key_cache_location": None,
            },
            RuntimeError,
            "Insecure key cache cannot be enabled without specifying its location",
        ),
        pytest.param(
            {"enable_unsafe_features": False, "simulate_encrypt_run_decrypt": True},
            RuntimeError,
            "Simulating encrypt/run/decrypt cannot be used without enabling unsafe features",
        ),
    ],
)
def test_configuration_bad_init(kwargs, expected_error, expected_message):
    """
    Test `__init__` method of `Configuration` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Configuration(**kwargs)

    assert str(excinfo.value) == expected_message


def test_configuration_fork():
    """
    Test `fork` method of `Configuration` class.
    """

    config1 = Configuration(enable_unsafe_features=True, loop_parallelize=False, p_error=0.1)
    config2 = config1.fork(enable_unsafe_features=False, loop_parallelize=True, p_error=None)

    assert config1 is not config2

    assert config1.enable_unsafe_features is True
    assert config1.loop_parallelize is False
    assert config1.p_error == 0.1

    assert config2.enable_unsafe_features is False
    assert config2.loop_parallelize is True
    assert config2.p_error is None


FORK_NAME = "fork" if sys.version_info < (3, 10) else "Configuration.fork"


@pytest.mark.parametrize(
    "kwargs,expected_error,expected_message",
    [
        pytest.param(
            {"foo": False},
            TypeError,
            f"{FORK_NAME}() got an unexpected keyword argument 'foo'",
        ),
        pytest.param(
            {"dump_artifacts_on_unexpected_failures": "yes"},
            TypeError,
            "Unexpected type for keyword argument 'dump_artifacts_on_unexpected_failures' "
            "(expected 'bool', got 'str')",
        ),
        pytest.param(
            {"insecure_key_cache_location": 3},
            TypeError,
            "Unexpected type for keyword argument 'insecure_key_cache_location' "
            "(expected 'Union[Path, str, NoneType]', got 'int')",
        ),
        pytest.param(
            {"p_error": "yes"},
            TypeError,
            "Unexpected type for keyword argument 'p_error' "
            "(expected 'Optional[float]', got 'str')",
        ),
        pytest.param(
            {"global_p_error": "mamma mia"},
            TypeError,
            "Unexpected type for keyword argument 'global_p_error' "
            "(expected 'Optional[float]', got 'str')",
        ),
        pytest.param(
            {"show_optimizer": "please"},
            TypeError,
            "Unexpected type for keyword argument 'show_optimizer' "
            "(expected 'Optional[bool]', got 'str')",
        ),
        pytest.param(
            {"parameter_selection_strategy": 42},
            TypeError,
            "42 cannot be parsed to a ParameterSelectionStrategy",
        ),
        pytest.param(
            {"parameter_selection_strategy": "bad"},
            ValueError,
            "'bad' is not a valid 'ParameterSelectionStrategy' (v0, mono, multi)",
        ),
        pytest.param(
            {"multi_parameter_strategy": 42},
            TypeError,
            "42 cannot be parsed to a MultiParameterStrategy",
        ),
        pytest.param(
            {"multi_parameter_strategy": "bad"},
            ValueError,
            "'bad' is not a valid 'MultiParameterStrategy' (precision, precision_and_norm2)",
        ),
        pytest.param(
            {"comparison_strategy_preference": 42},
            TypeError,
            "42 cannot be parsed to a ComparisonStrategy",
        ),
        pytest.param(
            {"comparison_strategy_preference": "bad"},
            ValueError,
            "'bad' is not a valid 'ComparisonStrategy' ("
            "one-tlu-promoted, "
            "three-tlu-casted, "
            "two-tlu-bigger-promoted-smaller-casted, "
            "two-tlu-bigger-casted-smaller-promoted, "
            "three-tlu-bigger-clipped-smaller-casted, "
            "two-tlu-bigger-clipped-smaller-promoted, "
            "chunked"
            ")",
        ),
        pytest.param(
            {"bitwise_strategy_preference": 42},
            TypeError,
            "42 cannot be parsed to a BitwiseStrategy",
        ),
        pytest.param(
            {"bitwise_strategy_preference": "bad"},
            ValueError,
            "'bad' is not a valid 'BitwiseStrategy' ("
            "one-tlu-promoted, "
            "three-tlu-casted, "
            "two-tlu-bigger-promoted-smaller-casted, "
            "two-tlu-bigger-casted-smaller-promoted, "
            "chunked"
            ")",
        ),
        pytest.param(
            {"multivariate_strategy_preference": 42},
            TypeError,
            "42 cannot be parsed to a MultivariateStrategy",
        ),
        pytest.param(
            {"multivariate_strategy_preference": "bad"},
            ValueError,
            "'bad' is not a valid 'MultivariateStrategy' (promoted, casted)",
        ),
        pytest.param(
            {"min_max_strategy_preference": 42},
            TypeError,
            "42 cannot be parsed to a MinMaxStrategy",
        ),
        pytest.param(
            {"min_max_strategy_preference": "bad"},
            ValueError,
            "'bad' is not a valid 'MinMaxStrategy' (one-tlu-promoted, three-tlu-casted, chunked)",
        ),
        pytest.param(
            {"additional_pre_processors": "bad"},
            TypeError,
            (
                "Unexpected type for keyword argument 'additional_pre_processors' "
                "(expected 'Optional[list[GraphProcessor]]', got 'str')"
            ),
        ),
        pytest.param(
            {"additional_post_processors": "bad"},
            TypeError,
            (
                "Unexpected type for keyword argument 'additional_post_processors' "
                "(expected 'Optional[list[GraphProcessor]]', got 'str')"
            ),
        ),
    ],
)
def test_configuration_bad_fork(kwargs, expected_error, expected_message):
    """
    Test `fork` method of `Configuration` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Configuration().fork(**kwargs)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "function,encryption_status,inputset,"
    "expected_bit_width_constraints,expected_bit_width_assignment",
    [
        pytest.param(
            lambda x, y: (x**2) + y,
            {"x": "encrypted", "y": "encrypted"},
            [(5, 120)],
            """

%0:
    <lambda>.%0 >= 3
%1:
    <lambda>.%1 >= 7
%2:
    <lambda>.%2 >= 2
%3:
    <lambda>.%3 >= 5
%4:
    <lambda>.%4 >= 8
    <lambda>.%3 == <lambda>.%1
    <lambda>.%1 == <lambda>.%4

            """,
            (
                """

 <lambda>.%0 = 3
 <lambda>.%1 = 8
 <lambda>.%2 = 2
 <lambda>.%3 = 8
 <lambda>.%4 = 8
<lambda>.max = 8

            """
                if USE_MULTI_PRECISION
                else """

 <lambda>.%0 = 8
 <lambda>.%1 = 8
 <lambda>.%2 = 8
 <lambda>.%3 = 8
 <lambda>.%4 = 8
<lambda>.max = 8

            """
            ),
        ),
    ],
)
def test_configuration_show_bit_width_constraints_and_assignment(
    function,
    encryption_status,
    inputset,
    expected_bit_width_constraints,
    expected_bit_width_assignment,
    helpers,
    capsys,
    monkeypatch,
):
    """
    Test compiling with configuration where show_bit_width_(constraints/assignments)=True.
    """

    monkeypatch.setattr("concrete.fhe.compilation.artifacts.get_terminal_size", lambda: 80)

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, encryption_status)
    compiler.compile(
        inputset,
        configuration.fork(show_bit_width_constraints=True, show_bit_width_assignments=True),
    )

    captured = capsys.readouterr()
    helpers.check_str(
        captured.out.strip(),
        f"""

Bit-Width Constraints for <lambda>
--------------------------------------------------------------------------------
{expected_bit_width_constraints.lstrip(os.linesep).rstrip()}
--------------------------------------------------------------------------------

Bit-Width Assignments for <lambda>
--------------------------------------------------------------------------------
{expected_bit_width_assignment.lstrip(os.linesep).rstrip()}
--------------------------------------------------------------------------------

        """.strip(),
    )


def test_set_security_level():
    """
    Test that the security level is set correctly.
    """

    # pylint: disable=no-self-argument,missing-class-docstring,missing-function-docstring
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

    # pylint: enable=no-self-argument,missing-class-docstring,missing-function-docstring

    inputset = [
        (np.random.randint(1, 20, size=()), np.random.randint(1, 20, size=())) for _ in range(100)
    ]

    # pylint: disable=no-member
    module1 = Module.compile(
        {"inc": inputset},
        security_level=SecurityLevel.SECURITY_128_BITS,
    )

    module2 = Module.compile(
        {"inc": inputset},
        security_level=SecurityLevel.SECURITY_128_BITS,
    )

    module3 = Module.compile(
        {"inc": inputset},
        security_level=SecurityLevel.SECURITY_132_BITS,
    )
    # pylint: enable=no-member

    assert (
        module1.server.client_specs.program_info.get_keyset_info()
        == module2.server.client_specs.program_info.get_keyset_info()
    )
    assert (
        module1.server.client_specs.program_info.get_keyset_info()
        != module3.server.client_specs.program_info.get_keyset_info()
    )
