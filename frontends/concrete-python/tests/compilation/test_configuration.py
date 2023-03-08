"""
Tests of `Configuration` class.
"""

import pytest

from concrete.numpy.compilation import Configuration


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

    config1 = Configuration(enable_unsafe_features=True, loop_parallelize=False)
    config2 = config1.fork(enable_unsafe_features=False, loop_parallelize=True)

    assert config1 is not config2

    assert config1.enable_unsafe_features is True
    assert config1.loop_parallelize is False

    assert config2.enable_unsafe_features is False
    assert config2.loop_parallelize is True


@pytest.mark.parametrize(
    "kwargs,expected_error,expected_message",
    [
        pytest.param(
            {"foo": False},
            TypeError,
            "Unexpected keyword argument 'foo'",
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
            "(expected 'Optional[str]', got 'int')",
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
    ],
)
def test_configuration_bad_fork(kwargs, expected_error, expected_message):
    """
    Test `fork` method of `Configuration` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Configuration().fork(**kwargs)

    assert str(excinfo.value) == expected_message
