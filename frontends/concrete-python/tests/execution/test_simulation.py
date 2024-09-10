"""
Tests execution of simulation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize("p_error", [0.5, 0.01, 0.005])
def test_tlu_error_rate(p_error):
    """
    Test that error rates are realistic.
    """
    repetition = max(1000, int(10 / p_error))
    assert repetition < 10_0000, "Test will be too long"
    conf = fhe.Configuration(p_error=p_error)

    precision = 3

    @fhe.compiler({"x": "encrypted"})
    def tlu(x):
        return fhe.univariate(lambda x: x)(x)

    inputset = [np.array(list(range(0, 2**precision)))]

    circuit = tlu.compile(inputset, conf)

    nok_fhe = 0
    nok_sim = 0
    size = 0
    for t in inputset * repetition:
        fhe_value = circuit.encrypt_run_decrypt(t)
        sim_value = circuit.simulate(t)
        for v, fhe_v, sim_v in zip(t, fhe_value, sim_value):
            size += 1
            nok_fhe += v != fhe_v
            nok_sim += v != sim_v

    fhe_p_error = nok_fhe / size
    sim_p_error = nok_sim / size

    assert p_error * size > 10  # test invalid if no more than 10 expected errors
    assert fhe_p_error
    assert sim_p_error
    # the sample size is not big => we accept significant releative error
    assert fhe_p_error == pytest.approx(p_error, rel=0.75)
    assert sim_p_error == pytest.approx(p_error, rel=0.75)
    assert fhe_p_error == pytest.approx(sim_p_error, rel=0.75)


@pytest.mark.parametrize("p_error", [0.01, 1e-120])
def test_approx_tlu_error_rate(p_error):
    """
    Test that error rates are realistic in approximate mode.
    """
    conf = fhe.Configuration(p_error=p_error)
    precision = 8
    lsbs_to_remove = 5

    def rnd(x):
        return (x + 2 ** (lsbs_to_remove - 1)) // 2**lsbs_to_remove

    @fhe.compiler({"x": "encrypted"})
    def tlu(x):
        rounded = fhe.round_bit_pattern(
            x,
            lsbs_to_remove=lsbs_to_remove,
            overflow_protection=False,
            exactness=fhe.Exactness.APPROXIMATE,
        )
        return fhe.univariate(rnd)(rounded)

    inputset = [np.array(list(range(0, 2**precision - 2**lsbs_to_remove)))]

    circuit = tlu.compile(inputset, conf)

    nok_fhe = 0
    nok_sim = 0
    size = 0
    for t in inputset * 30:
        fhe_values = circuit.encrypt_run_decrypt(t)
        sim_values = circuit.simulate(t)
        for x, fhe_v, sim_v in zip(t, fhe_values, sim_values):
            r = rnd(x)
            size += 1
            nok_fhe += r != fhe_v
            nok_sim += r != sim_v
            print(r, fhe_v, sim_v)

    fhe_p_error = nok_fhe / size
    sim_p_error = nok_sim / size

    # error rate is always significant (>3%) in approximate mode
    assert p_error + 0.005 < fhe_p_error
    assert p_error + 0.005 < sim_p_error
    assert fhe_p_error < 0.2 + p_error
    assert sim_p_error < 0.2 + p_error

    assert fhe_p_error == pytest.approx(sim_p_error, rel=0.75)
