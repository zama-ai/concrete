# -*- coding: utf-8 -*-
from sage.all import binomial, ZZ, log, ceil, RealField, oo, exp, pi
from sage.all import RealDistribution, RR, sqrt, prod, erf
from .nd import sigmaf


def mitm_babai_probability(r, stddev, q, fast=False):
    """
    Compute the "e-admissibility" probability associated to the mitm step, according to
    [EPRINT:SonChe19]_

    :params r: the squared GSO lengths
    :params stddev: the std.dev of the error distribution
    :params q: the LWE modulus
    :param fast: toggle for setting p = 1 (faster, but underestimates security)
    :return: probability for the mitm process

    # NOTE: the model sometimes outputs negative probabilities, we set p = 0 in this case
    """

    if fast:
        # overestimate the probability -> underestimate security
        p = 1
    else:
        # get non-squared norms
        R = [sqrt(s) for s in r]
        alphaq = sigmaf(stddev)
        probs = [
            RR(
                erf(s * sqrt(RR(pi)) / alphaq)
                + (alphaq / s) * ((exp(-s * sqrt(RR(pi)) / alphaq) - 1) / RR(pi))
            )
            for s in R
        ]
        p = RR(prod(probs))
        if p < 0 or p > 1:
            p = 0.0
    return p


def babai(r, norm):
    """
    Babai probability following [EPRINT:Wun16]_.

    """
    R = [RR(sqrt(t) / (2 * norm)) for t in r]
    T = RealDistribution("beta", ((len(r) - 1) / 2, 1.0 / 2))
    probs = [1 - T.cum_distribution_function(1 - s ** 2) for s in R]
    return prod(probs)


def drop(n, h, k, fail=0, rotations=False):
    """
    Probability that ``k`` randomly sampled components have ``fail`` non-zero components amongst
    them.

    :param n: LWE dimension `n > 0`
    :param h: number of non-zero components
    :param k: number of components to ignore
    :param fail: we tolerate ``fail`` number of non-zero components amongst the `k` ignored
        components
    :param rotations: consider rotations of the basis to exploit ring structure (NTRU only)
    """

    N = n  # population size
    K = n - h  # number of success states in the population
    n = k  # number of draws
    k = n - fail  # number of observed successes
    prob_drop = binomial(K, k) * binomial(N - K, n - k) / binomial(N, n)
    if rotations:
        return 1 - (1 - prob_drop) ** N
    else:
        return prob_drop


def amplify(target_success_probability, success_probability, majority=False):
    """
    Return the number of trials needed to amplify current `success_probability` to
    `target_success_probability`

    :param target_success_probability: targeted success probability < 1
    :param success_probability: targeted success probability < 1
    :param majority: if `True` amplify a deicsional problem, not a computational one
       if `False` then we assume that we can check solutions, so one success suffices

    :returns: number of required trials to amplify
    """
    if target_success_probability < success_probability:
        return ZZ(1)
    if success_probability == 0.0:
        return oo

    prec = max(
        53,
        2 * ceil(abs(log(success_probability, 2))),
        2 * ceil(abs(log(1 - success_probability, 2))),
        2 * ceil(abs(log(target_success_probability, 2))),
        2 * ceil(abs(log(1 - target_success_probability, 2))),
    )
    prec = min(prec, 2048)
    RR = RealField(prec)

    success_probability = RR(success_probability)
    target_success_probability = RR(target_success_probability)

    try:
        if majority:
            eps = success_probability / 2
            return ceil(2 * log(2 - 2 * target_success_probability) / log(1 - 4 * eps ** 2))
        else:
            # target_success_probability = 1 - (1-success_probability)^trials
            return ceil(log(1 - target_success_probability) / log(1 - success_probability))
    except ValueError:
        return oo


def amplify_sigma(target_advantage, sigma, q):
    """
    Amplify distinguishing advantage for a given σ and q

    :param target_advantage:
    :param sigma: (Lists of) Gaussian width parameters
    :param q: Modulus q > 0

    """
    try:
        sigma = sum(sigma_ ** 2 for sigma_ in sigma).sqrt()
    except TypeError:
        pass
    advantage = float(exp(-float(pi) * (float(sigma / q) ** 2)))
    return amplify(target_advantage, advantage, majority=True)
