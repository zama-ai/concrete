# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using Gröbner bases.

See :ref:`Arora-GB` for an overview.

"""
from sage.all import (
    PowerSeriesRing,
    QQ,
    RR,
    oo,
    binomial,
    sqrt,
    ceil,
    floor,
    exp,
    log,
    pi,
    RealField,
)
from .cost import Cost
from .lwe_parameters import LWEParameters
from .io import Logging


def gb_cost(n, D, omega=2, prec=None):
    """
    Estimate the complexity of computing a Gröbner basis.

    :param n: Number of variables n > 0.
    :param D: Tuple of `(d,m)` pairs where `m` is number polynomials and `d` is a degree.
    :param omega: Linear algebra exponent, i.e. matrix-multiplication costs `O(n^ω)` operations.
    :param prec: Compute power series up to this precision (default: `2n`).

    EXAMPLE::

        >>> from estimator.gb import gb_cost
        >>> gb_cost(128, [(2, 256)])
        rop: ≈2^144.6, dreg: 17, mem: ≈2^144.6

    """
    prec = 2 * n if prec is None else prec

    R = PowerSeriesRing(QQ, "z", prec)
    z = R.gen()
    z = z.add_bigoh(prec)
    s = R(1)
    s = s.add_bigoh(prec)

    for d, m in D:
        s *= (1 - z ** d) ** m
    s /= (1 - z) ** n

    retval = Cost(rop=oo, dreg=oo)
    retval.register_impermanent({"rop": True, "dreg": False, "mem": False})

    for dreg in range(prec):
        if s[dreg] < 0:
            break
    else:
        return retval

    retval["dreg"] = dreg
    retval["rop"] = binomial(n + dreg, dreg) ** omega
    retval["mem"] = binomial(n + dreg, dreg) ** 2

    return retval


class AroraGB:
    @staticmethod
    def ps_single(C):
        """
        Probability that a Gaussian is within `C` standard deviations.
        """
        RR = RealField(256)
        C = RR(C)
        return RR(1 - (RR(2) / (C * RR(sqrt(2 * pi))) * exp(-(C ** 2) / RR(2))))  # noqa

    @classmethod
    def cost_bounded(cls, params, success_probability=0.99, omega=2, log_level=1, **kwds):
        """
        Estimate cost using absolute bounds for secrets and noise.

        :param params: LWE parameters.
        :param success_probability: target success probability
        :param omega: linear algebra constant.

        """
        d = params.Xe.bounds[1] - params.Xe.bounds[0] + 1
        dn = cls.equations_for_secret(params)
        cost = gb_cost(params.n, [(d, params.m)] + dn)
        cost["t"] = (d - 1) // 2
        if cost["dreg"] < oo and binomial(params.n + cost["dreg"], cost["dreg"]) < params.m:
            cost["m"] = binomial(params.n + cost["dreg"], cost["dreg"])
        else:
            cost["m"] = params.m
        cost.register_impermanent(t=False, m=True)
        return cost

    @classmethod
    def cost_Gaussian_like(cls, params, success_probability=0.99, omega=2, log_level=1, **kwds):
        """
        Estimate cost using absolute bounds for secrets and Gaussian tail bounds for noise.

        :param params: LWE parameters.
        :param success_probability: target success probability
        :param omega: linear algebra constant.

        """
        dn = cls.equations_for_secret(params)

        best, stuck = None, 0
        for t in range(ceil(params.Xe.stddev), params.n):
            d = 2 * t + 1
            C = RR(t / params.Xe.stddev)
            assert C >= 1  # if C is too small, we ignore it
            # Pr[success]^m = Pr[overall success]
            single_prob = AroraGB.ps_single(C)
            m_req = log(success_probability, 2) / log(single_prob, 2)
            m_req = floor(m_req)

            if m_req > params.m:
                break

            current = gb_cost(params.n, [(d, m_req)] + dn, omega)

            if current["dreg"] == oo:
                continue

            current["t"] = t
            current["m"] = m_req
            current.register_impermanent(t=False, m=True)
            current = current.reorder("rop", "m", "dreg", "t")

            Logging.log("repeat", log_level + 1, f"{repr(current)}")

            if best is None:
                best = current
            else:
                if best > current:
                    best = current
                    stuck = 0
                else:
                    stuck += 1
                    if stuck >= 5:
                        break

        if best is None:
            best = Cost(rop=oo, dreg=oo)
        return best

    @classmethod
    def equations_for_secret(cls, params):
        """
        Return ``(d,n)`` tuple to encode that `n` equations of degree `d` are available from the LWE secret.

        :param params: LWE parameters.

        """
        if params.Xs <= params.Xe:
            a, b = params.Xs.bounds
            if b - a < oo:
                d = b - a + 1
            elif params.Xs.is_Gaussian_like:
                d = 2 * ceil(3 * params.Xs.stddev) + 1
            else:
                raise NotImplementedError(f"Do not know how to handle {params.Xs}.")
            dn = [(d, params.n)]
        else:
            dn = []
        return dn

    def __call__(
        self, params: LWEParameters, success_probability=0.99, omega=2, log_level=1, **kwds
    ):
        """
        Arora-GB as described in [ICALP:AroGe11]_, [EPRINT:ACFP14]_.

        :param params: LWE parameters.
        :param success_probability: targeted success probability < 1.
        :param omega: linear algebra constant.
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``m``: Number of samples consumed.
        - ``dreg``: The degree of regularity or "solving degree".
        - ``t``: Polynomials of degree 2t + 1 are considered.
        - ``mem``: Total memory usage.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=64, q=7681, Xs=ND.DiscreteGaussian(3.0), Xe=ND.DiscreteGaussian(3.0), m=2**50)
            >>> LWE.arora_gb(params)
            rop: ≈2^307.1, m: ≈2^46.8, dreg: 99, t: 25, mem: ≈2^307.1, tag: arora-gb

        TESTS::

            >>> LWE.arora_gb(params.updated(m=2**120))
            rop: ≈2^282.6, m: ≈2^101.1, dreg: 83, t: 36, mem: ≈2^282.6, tag: arora-gb
            >>> LWE.arora_gb(params.updated(Xe=ND.UniformMod(7)))
            rop: ≈2^60.6, dreg: 7, mem: ≈2^60.6, t: 3, m: ≈2^30.3, tag: arora-gb
            >>> LWE.arora_gb(params.updated(Xe=ND.CenteredBinomial(8)))
            rop: ≈2^122.3, dreg: 19, mem: ≈2^122.3, t: 8, m: ≈2^50.0, tag: arora-gb
            >>> LWE.arora_gb(params.updated(Xs=ND.UniformMod(5), Xe=ND.CenteredBinomial(4), m=1024))
            rop: ≈2^227.2, dreg: 54, mem: ≈2^227.2, t: 4, m: 1024, tag: arora-gb
            >>> LWE.arora_gb(params.updated(Xs=ND.UniformMod(3), Xe=ND.CenteredBinomial(4), m=1024))
            rop: ≈2^189.9, dreg: 39, mem: ≈2^189.9, t: 4, m: 1024, tag: arora-gb

        ..  [EPRINT:ACFP14] Martin R. Albrecht, Carlos Cid, Jean-Charles Faugère & Ludovic Perret. (2014).
            Algebraic algorithms for LWE. https://eprint.iacr.org/2014/1018

        ..  [ICALP:AroGe11] Sanjeev Aror & Rong Ge. (2011). New algorithms for learning in presence of
            errors.  In L.  Aceto, M.  Henzinger, & J.  Sgall, ICALP 2011, Part I (pp.  403–415).:
            Springer, Heidelberg.
        """
        params = params.normalize()

        best = Cost(rop=oo, dreg=oo)

        if params.Xe.is_bounded:
            cost = self.cost_bounded(
                params,
                success_probability=success_probability,
                omega=omega,
                log_level=log_level,
            )
            Logging.log("gb", log_level, f"b: {repr(cost)}")
            best = min(best, cost, key=lambda x: x["dreg"])

        if params.Xe.is_Gaussian_like:
            cost = self.cost_Gaussian_like(
                params,
                success_probability=success_probability,
                omega=omega,
                log_level=log_level,
            )
            Logging.log("gb", log_level, f"G: {repr(cost)}")
            best = min(best, cost, key=lambda x: x["dreg"])

        best["tag"] = "arora-gb"
        best["problem"] = params
        return best

    __name__ = "arora_gb"


arora_gb = AroraGB()
