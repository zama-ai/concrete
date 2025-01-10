"""
Tests of everything related to restrictions.
"""

import math
import numpy as np
import pytest
from mlir._mlir_libs._concretelang._compiler import (
    KeysetInfo,
    KeysetRestriction,
    InternalPartitionDefinition,
    ExternalPartitionDefinition,
    RangeRestriction,
)

from concrete import fhe
from concrete.fhe import tfhers

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


def test_generic_restriction_without_external():
    """
    Test that compiling a module works.
    """

    generic_keyset_info = KeysetInfo.generate_virtual(
        [InternalPartitionDefinition(8, 10.0), InternalPartitionDefinition(10, 10000.0)], []
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
    assert all([k in generic_keyset_info.secret_keys() for k in compiled_keyset_info.secret_keys()])


def test_generic_restriction_with_external(helpers):
    """
    Test that compiling a module works.
    """

    dtype = tfhers.TFHERSIntegerType(
        False,
        bit_width=8,
        carry_width=3,
        msg_width=2,
        params=tfhers.CryptoParams(
            lwe_dimension=909,
            glwe_dimension=1,
            polynomial_size=4096,
            pbs_base_log=15,
            pbs_level=2,
            lwe_noise_distribution=0,
            glwe_noise_distribution=2.168404344971009e-19,
            encryption_key_choice=tfhers.EncryptionKeyChoice.BIG,
        ),
    )
    config = helpers.configuration()
    options = config.to_compilation_options().get_optimizer_options()

    generic_keyset_info = KeysetInfo.generate_virtual(
        [
            InternalPartitionDefinition(3, 10.0),
            InternalPartitionDefinition(5, 10.0)
        ],
        [
            dtype.external_partition_definition()
        ],
        options
    )

    parameters = {
        "x": {"range": [0, 2**7], "status": "encrypted"},
        "y": {"range": [0, 2**7], "status": "encrypted"},
    }

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    def binary_tfhers(x, y, binary_op, tfhers_type):
        """wrap binary op in tfhers conversion (2 tfhers inputs)"""
        x = tfhers.to_native(x)
        y = tfhers.to_native(y)
        return tfhers.from_native(binary_op(x, y), tfhers_type)

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, lambda x, y: x + y, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]

    circuit = compiler.compile(inputset, config, keyset_restriction=generic_keyset_info.get_restriction())

    compiled_keyset_info = circuit._module.keys.specs.program_info.get_keyset_info()
    assert all([k in generic_keyset_info.secret_keys() for k in compiled_keyset_info.secret_keys()])
