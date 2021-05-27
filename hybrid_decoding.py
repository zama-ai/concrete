# -*- coding: utf-8 -*-
"""
Seucrity estimates for the Hybrid Decoding attack

Requires a local copy of the LWE Estimator from https://bitbucket.org/malb/lwe-estimator/src/master/estimator.py in a folder called "estimator"
"""

from sage.all import ZZ, binomial, sqrt, log, exp, oo, pi, prod, RR
from sage.probability.probability_distribution import RealDistribution
from estimator import estimator as est
from concrete_params import concrete_LWE_params

## Core cost models

core_sieve = lambda beta, d, B: ZZ(2)**RR(0.292*beta + 16.4)
core_qsieve = lambda beta, d, B: ZZ(2)**RR(0.265*beta + 16.4)

## Utility functions

def sq_GSO(d, beta, det):
    """
    Return squared GSO lengths after lattice reduction according to the GSA

    :param q: LWE modulus
    :param d: lattice dimension
    :param beta: blocksize used in BKZ
    :param det: lattice determinant

    """

    r = []
    for i in range(d):
        r_i = est.delta_0f(beta)**(((-2*d*i) / (d-1))+d) * det**(1/d)
        r.append(r_i**2)

    return r


def babai_probability_wun16(r, norm):
     """
     Compute the probability of Babai's Nearest Plane, using techniques from the NTRULPrime submission to NIST

     :param r: squared GSO lengths
     :param norm: expected norm of the target vector

     """
     R = [RR(sqrt(t)/(2*norm)) for t in r]
     T = RealDistribution('beta', ((len(r)-1)/2,1./2))
     probs = [1 - T.cum_distribution_function(1 - s**2) for s in R]
     return prod(probs)


## Estimate hybrid decoding attack complexity

def hybrid_decoding_attack(n, alpha, q, m, secret_distribution,
                   beta, tau = None, mitm=True, reduction_cost_model=est.BKZ.sieve):
    """
    Estimate cost of the Hybrid Attack,

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/sqrt{2π}`
    :param q: modulus `0 < q`
    :param m: number of LWE samples `m > 0`
    :param secret_distribution: distribution of secret
    :param beta: BKZ block size β
    :param tau: guessing dimension τ
    :param mitm: simulate MITM approach (√ of search space)
    :param reduction_cost_model: BKZ reduction cost model

    EXAMPLE:

    hybrid_decoding_attack(beta = 100, tau = 250, mitm = True, reduction_cost_model = est.BKZ.sieve, **example_64())

         rop:   2^65.1
         pre:   2^64.8
        enum:   2^62.5
        beta:      100
         |S|:   2^73.1
        prob: 0.104533
       scale:   12.760
          pp:       11
           d:     1798
      repeat:       42

    """

    n, alpha, q = est.Param.preprocess(n, alpha, q)

    # d is the dimension of the attack lattice
    d = m + n - tau

    # h is the Hamming weight of the secret
    # NOTE: binary secrets are assumed to have Hamming weight ~n/2, ternary secrets ~2n/3
    # this aligns with the assumptions made in the LWE Estimator
    h = est.SDis.nonzero(secret_distribution, n=n)
    sd = alpha*q/sqrt(2*pi)

    # compute the scaling factor used in the primal lattice to balance the secret and error
    scale = est._primal_scale_factor(secret_distribution, alpha=alpha, q=q, n=n)

    # 1. get squared-GSO lengths via the Geometric Series Assumption
    # we could also consider using the BKZ simulator, using the GSA is conservative
    r = sq_GSO(d, beta, q**m * scale**(n-tau))

    # 2. Costs
    bkz_cost = est.lattice_reduction_cost(reduction_cost_model, est.delta_0f(beta), d)
    enm_cost = est.Cost()
    enm_cost["rop"] = d**2/(2**1.06)

    # 3. Size of search space
    # We need to do one BDD call at least
    search_space, prob, hw = ZZ(1), 1.0, 0

    # if mitm is True, sqrt speedup in the guessing phase. This allows us to square the size
    # of the search space at no extra cost.
    # NOTE: we conservatively assume that this mitm process succeeds with probability 1.
    ssf = sqrt if mitm else lambda x: x

    # use the secret distribution bounds to determine the size of the search space
    a, b = est.SDis.bounds(secret_distribution)

    # perform "searching". This part of the code balances the enm_cost with the cost of lattice
    # reduction, where enm_cost is the total cost of calling Babai's algorithm on each vector in
    # the search space.

    if tau:
        prob = est.success_probability_drop(n, h, tau)
        hw = 1
        while hw < h and hw < tau:
            prob += est.success_probability_drop(n, h, tau, fail=hw)
            search_space += binomial(tau, hw) * (b-a)**hw

            if enm_cost.repeat(ssf(search_space))["rop"] > bkz_cost["rop"]:
                # we moved too far, so undo
                prob -= est.success_probability_drop(n, h, tau, fail=hw)
                search_space -= binomial(tau, hw) * (b-a)**hw
                hw -= 1
                break
            hw += 1

        enm_cost = enm_cost.repeat(ssf(search_space))

    # we use the expectation of the target norm. This could be longer, or shorter, for any given instance.
    target_norm = sqrt(m * sd**2 + h * RR((n-tau)/n) * scale**2)

    # account for the success probability of Babai's algorithm
    prob*=babai_probability_wun16(r, target_norm)

    # create a cost string, as in the LWE Estimator, to store the attack parameters and costs
    ret = est.Cost()
    ret["rop"] = bkz_cost["rop"] + enm_cost["rop"]
    ret["pre"] = bkz_cost["rop"]
    ret["enum"] = enm_cost["rop"]
    ret["beta"] = beta
    ret["|S|"] = search_space
    ret["prob"] = prob
    ret["scale"] = scale
    ret["pp"] = hw
    ret["d"] = d
    ret["tau"] = tau

    # 5. Repeat whole experiment ~1/prob times
    ret = ret.repeat(est.amplify(0.99, prob), select={"rop": True,
                                                      "pre": True,
                                                      "enum": True,
                                                      "beta": False,
                                                      "d": False,
                                                      "|S|": False,
                                                      "scale": False,
                                                      "prob": False,
                                                      "pp": False,
                                                      "tau": False})

    return ret


## Optimize attack parameters

def parameter_search(n, alpha, q, m, secret_distribution, mitm = True, reduction_cost_model=est.BKZ.sieve):

    """
    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/sqrt{2π}`
    :param q: modulus `0 < q`
    :param m: number of LWE samples `m > 0`
    :param secret_distribution: distribution of secret
    :param beta_search: tuple (β_min,  β_max, granularity) for the search space of β, default is (60,301,20)
    :param tau: tuple (τ_min, τ_max, granularity) for the search space of τ, default is (0,501,20)
    :param mitm: simulate MITM approach (√ of search space)
    :param reduction_cost_model: BKZ reduction cost model

    EXAMPLE:

    parameter_search(mitm = False, reduction_cost_model = est.BKZ.sieve, **example_64())

         rop:   2^69.5
         pre:   2^68.9
        enum:   2^68.0
        beta:      110
         |S|:   2^40.9
        prob: 0.045060
       scale:   12.760
          pp:        6
           d:     1730
      repeat:      100
         tau:      170

    parameter_search(mitm = True, reduction_cost_model = est.BKZ.sieve, **example_64())

         rop:   2^63.4
         pre:   2^63.0
        enum:   2^61.5
        beta:       95
         |S|:   2^72.0
        prob: 0.125126
       scale:   12.760
          pp:       11
           d:     1666
      repeat:       35
         tau:      234

    """

    primald = est.partial(est.drop_and_solve, est.dual_scale, postprocess=True, decision=True)
    bl = primald(n, alpha, q, secret_distribution=secret_distribution, m=m, reduction_cost_model=reduction_cost_model)

    # we take the number of LWE samples used to be the same as in the primal attack in the LWE Estimator
    m = bl["m"]

    f = est.partial(hybrid_decoding_attack, n=n, alpha=alpha, q=q, m=m, secret_distribution=secret_distribution,
                    reduction_cost_model=reduction_cost_model,
                    mitm=mitm)

    # NOTE: we decribe our searching strategy below. To produce more accurate estimates,
    # change this part of the code to ensure a more granular search. As we are using
    # homomorphic-encryption style parameters, the running time of the code can be quite high,
    # justifying the below choices.
    # We start at beta = 60 and go up to beta_max in steps of 50

    beta_max = bl["beta"] + 100
    beta_search = (40, beta_max, 50)

    best = None
    for beta in range(beta_search[0], beta_search[1], beta_search[2])[::-1]:
        tau = 0
        best_beta = None
        count = 3
        while tau < n:
            if count >= 0:
                cost = f(beta=beta, tau=tau)
                if best_beta is not None:
                    # if two consecutive estimates don't decrease, stop optimising over tau
                    if best_beta["rop"] < cost["rop"]:
                        count -= 1
                cost["tau"] = tau
                if best_beta is None:
                    best_beta = cost
                if RR(log(cost["rop"],2)) < RR(log(best_beta["rop"],2)):
                    best_beta = cost
                if best is None:
                    best = cost
                if RR(log(cost["rop"],2)) < RR(log(best["rop"],2)):
                    best = cost
            tau += n//100

    # now do a second, more granular search
    # we start at the beta which produced the lowest running time, and search ± 25 in steps of 10
    tau_gap = max(n//100, 1)
    for beta in range(best["beta"] - 25, best["beta"] + 25, 10)[::-1]:
        tau = max(best["tau"] - 25,0)
        best_beta = None
        count = 3
        while tau <= best["tau"] + 25:
            if count >= 0:
                cost = f(beta=beta, tau=tau)
                if best_beta is not None:
                    # if two consecutive estimates don't decrease, stop optimising over tau
                    if best_beta["rop"] < cost["rop"]:
                        count -= 1
                cost["tau"] = tau
                if best_beta is None:
                    best_beta = cost
                if RR(log(cost["rop"],2)) < RR(log(best_beta["rop"],2)):
                    best_beta = cost
                if best is None:
                    best = cost
                if RR(log(cost["rop"],2)) < RR(log(best["rop"],2)):
                    best = cost
            tau += tau_gap

    return best

def get_all_security_levels(params):
    """ A function which gets the security levels of a collection of TFHE parameters,
    using the four cost models: classical, quantum, classical_conservative, and
    quantum_conservative
    :param params: a dictionary of LWE parameter sets (see concrete_params)

    EXAMPLE:
    sage: X = get_all_security_levels(concrete_LWE_params)
    sage: X
    [['LWE128_256',
    126.692189756144,
    117.566189756144,
    98.6960000000000,
    89.5700000000000], ...]
    """

    RESULTS = []

    for param in params:

        results = [param]
        x = params["{}".format(param)]
        n = x["n"] * x["k"]
        q = 2 ** 32
        sd = 2 ** (x["sd"]) * q
        alpha = sqrt(2 * pi) * sd / RR(q)
        secret_distribution = (0, 1)
        # assume access to an infinite number of papers
        m = oo

        model = est.BKZ.sieve
        estimate = parameter_search(mitm = True, reduction_cost_model = est.BKZ.sieve, n = n, q = q, alpha = alpha, m = m, secret_distribution = secret_distribution)
        results.append(get_security_level(estimate))

        RESULTS.append(results)

    return RESULTS