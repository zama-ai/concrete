"""
Tests of everything related to restrictions.
"""

import numpy as np

# pylint: disable=import-error
from mlir._mlir_libs._concretelang._compiler import (
    KeysetInfo,
    PartitionDefinition,
    RangeRestriction,
)

from concrete import fhe

# pylint: enable=import-error


# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-argument, unused-variable, no-member, unused-argument, function-redefined, expression-not-assigned
# same disables for ruff:
# ruff: noqa: N805, E501, F841, ARG002, F811, B015


def test_range_restriction():
    """
    Test that compiling a module works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    range_restriction = RangeRestriction()
    internal_lwe_dimension = 999
    range_restriction.add_available_internal_lwe_dimension(internal_lwe_dimension)
    glwe_log_polynomial_size = 12
    range_restriction.add_available_glwe_log_polynomial_size(glwe_log_polynomial_size)
    glwe_dimension = 2
    range_restriction.add_available_glwe_dimension(glwe_dimension)
    pbs_level_count = 3
    range_restriction.add_available_pbs_level_count(pbs_level_count)
    pbs_base_log = 11
    range_restriction.add_available_pbs_base_log(pbs_base_log)
    ks_level_count = 3
    range_restriction.add_available_ks_level_count(ks_level_count)
    ks_base_log = 6
    range_restriction.add_available_ks_base_log(ks_base_log)
    module = Module.compile({"inc": inputset}, range_restriction=range_restriction)
    keyset_info = module.keys.specs.program_info.get_keyset_info()
    assert keyset_info.bootstrap_keys()[0].polynomial_size() == 2**glwe_log_polynomial_size
    assert keyset_info.bootstrap_keys()[0].input_lwe_dimension() == internal_lwe_dimension
    assert keyset_info.bootstrap_keys()[0].glwe_dimension() == glwe_dimension
    assert keyset_info.bootstrap_keys()[0].level() == pbs_level_count
    assert keyset_info.bootstrap_keys()[0].base_log() == pbs_base_log
    assert keyset_info.keyswitch_keys()[0].level() == ks_level_count
    assert keyset_info.keyswitch_keys()[0].base_log() == ks_base_log
    assert keyset_info.secret_keys()[0].dimension() == 2**glwe_log_polynomial_size * glwe_dimension
    assert keyset_info.secret_keys()[1].dimension() == internal_lwe_dimension


def test_keyset_restriction():
    """
    Test that compiling a module works.
    """

    @fhe.module()
    class Big:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 200

    big_inputset = [np.random.randint(1, 200, size=()) for _ in range(100)]

    @fhe.module()
    class Small:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 20

    small_inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]

    big_module = Big.compile(
        {"inc": big_inputset},
    )
    big_keyset_info = big_module.keys.specs.program_info.get_keyset_info()

    small_module = Small.compile(
        {"inc": small_inputset},
    )
    small_keyset_info = small_module.keys.specs.program_info.get_keyset_info()
    assert big_keyset_info != small_keyset_info

    restriction = big_keyset_info.get_restriction()
    restricted_module = Small.compile({"inc": small_inputset}, keyset_restriction=restriction)
    restricted_keyset_info = restricted_module.keys.specs.program_info.get_keyset_info()
    assert big_keyset_info == restricted_keyset_info
    assert small_keyset_info != restricted_keyset_info


def test_generic_restriction():
    """
    Test that compiling a module works.
    """

    generic_keyset_info = KeysetInfo.generate_virtual(
        [PartitionDefinition(8, 10.0), PartitionDefinition(10, 10000.0)], True
    )

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return (x + 1) % 200

    inputset = [np.random.randint(1, 200, size=()) for _ in range(100)]
    restricted_module = Module.compile(
        {"inc": inputset},
        keyset_restriction=generic_keyset_info.get_restriction(),
    )
    compiled_keyset_info = restricted_module.keys.specs.program_info.get_keyset_info()
    assert all(k in generic_keyset_info.secret_keys() for k in compiled_keyset_info.secret_keys())
