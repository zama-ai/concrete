# -*- coding: utf-8 -*-
"""
See :ref:`Coded-BKW for LWE` for what is available.
"""
from sage.all import ceil, log, floor, sqrt, var, find_root, erf, oo
from .lwe_parameters import LWEParameters
from .util import local_minimum
from .cost import Cost
from .errors import InsufficientSamplesError
from .prob import amplify_sigma
from .nd import sigmaf
from .io import Logging

cfft = 1  # convolutions mod q


class CodedBKW:
    @staticmethod
    def N(i, sigma_set, b, q):
        """
        Return `N_i` for the `i`-th `[N_i, b]` linear code.

        :param i: index
        :param sigma_set: target noise level
        """
        return floor(b / (1 - log(12 * sigma_set ** 2 / 2 ** i, q) / 2))

    @staticmethod
    def ntest(n, ell, t1, t2, b, q):  # noqa
        """
        If the parameter ``ntest`` is not provided, we use this function to estimate it.

        :param n: LWE dimension > 0.
        :param ell: Table size for hypothesis testing.
        :param t1: Number of normal BKW steps.
        :param t2: Number of coded BKW steps.
        :param b: Table size for BKW steps.

        """

        # there is no hypothesis testing because we have enough normal BKW
        # tables to cover all of of n
        if t1 * b >= n:
            return 0

        # solve for nest by aiming for ntop == 0
        ntest = var("ntest")
        sigma_set = sqrt(q ** (2 * (1 - ell / ntest)) / 12)
        ncod = sum([CodedBKW.N(i, sigma_set, b, q) for i in range(1, t2 + 1)])
        ntop = n - ncod - ntest - t1 * b

        try:
            start = max(int(round(find_root(ntop, 2, n - t1 * b + 1, rtol=0.1))) - 1, 2)
        except RuntimeError:
            start = 2
        ntest_min = 1
        for ntest in range(start, n - t1 * b + 1):
            if abs(ntop(ntest=ntest).n()) < abs(ntop(ntest=ntest_min).n()):
                ntest_min = ntest
            else:
                break
        return int(ntest_min)

    def t1(params: LWEParameters, ell, t2, b, ntest=None):
        """
        We compute t1 from N_i by observing that any N_i ≤ b gives no advantage over vanilla
        BKW, but the estimates for coded BKW always assume quantisation noise, which is too
        pessimistic for N_i ≤ b.
        """

        t1 = 0
        if ntest is None:
            ntest = CodedBKW.ntest(params.n, ell, t1, t2, b, params.q)
        sigma_set = sqrt(params.q ** (2 * (1 - ell / ntest)) / 12)
        Ni = [CodedBKW.N(i, sigma_set, b, params.q) for i in range(1, t2 + 1)]
        t1 = len([e for e in Ni if e <= b])
        # there is no point in having more tables than needed to cover n
        if b * t1 > params.n:
            t1 = params.n // b
        return t1

    @staticmethod
    def cost(
        t2: int,
        b: int,
        ntest: int,
        params: LWEParameters,
        success_probability=0.99,
        log_level=1,
    ):
        """
        Coded-BKW cost.

        :param t2: Number of coded BKW steps (≥ 0).
        :param b: Table size (≥ 1).
        :param success_probability: Targeted success probability < 1.
        :param ntest: Number of coordinates to hypothesis test.

        """
        cost = Cost()

        # Our cost is mainly determined by q^b, on the other hand there are expressions in q^(ℓ+1)
        # below, hence, we set ℓ = b - 1. This allows to achieve the performance reported in
        # [C:GuoJohSta15].

        cost["b"] = b
        ell = b - 1  # noqa
        cost["ell"] = ell

        secret_bounds = params.Xs.bounds
        if params.Xs.is_Gaussian_like and params.Xs.mean == 0:
            secret_bounds = (
                max(secret_bounds[0], -3 * params.Xs.stddev),
                min(secret_bounds[1], 3 * params.Xs.stddev),
            )

        # base of the table size.
        zeta = secret_bounds[1] - secret_bounds[0] + 1

        t1 = CodedBKW.t1(params, ell, t2, b, ntest)
        t2 -= t1

        cost["t1"] = t1
        cost["t2"] = t2

        cost.register_impermanent(t1=False, t2=False)

        # compute ntest with the t1 just computed
        if ntest is None:
            ntest = CodedBKW.ntest(params.n, ell, t1, t2, b, params.q)

        # if there's no ntest then there's no `σ_{set}` and hence no ncod
        if ntest:
            sigma_set = sqrt(params.q ** (2 * (1 - ell / ntest)) / 12)
            ncod = sum([CodedBKW.N(i, sigma_set, b, params.q) for i in range(1, t2 + 1)])
        else:
            ncod = 0

        ntot = ncod + ntest
        ntop = max(params.n - ncod - ntest - t1 * b, 0)
        cost["#cod"] = ncod  # coding step
        cost["#top"] = ntop  # guessing step, typically zero
        cost["#test"] = ntest  # hypothesis testing

        cost.register_impermanent({"#cod": False, "#top": False, "#test": False})

        # Theorem 1: quantization noise + addition noise
        coding_variance = params.Xs.stddev ** 2 * sigma_set ** 2 * ntot
        sigma_final = float(sqrt(2 ** (t1 + t2) * params.Xe.stddev ** 2 + coding_variance))

        M = amplify_sigma(success_probability, sigmaf(sigma_final), params.q)
        if M is oo:
            cost["rop"] = oo
            cost["m"] = oo
            return cost
        m = (t1 + t2) * (params.q ** b - 1) / 2 + M
        cost["m"] = float(m)
        cost.register_impermanent(m=True)

        if not params.Xs <= params.Xe:
            # Equation (7)
            n = params.n - t1 * b
            C0 = (m - n) * (params.n + 1) * ceil(n / (b - 1))
            assert C0 >= 0
        else:
            C0 = 0

        # Equation (8)
        C1 = sum(
            [(params.n + 1 - i * b) * (m - i * (params.q ** b - 1) / 2) for i in range(1, t1 + 1)]
        )
        assert C1 >= 0

        # Equation (9)
        C2_ = sum(
            [
                4 * (M + i * (params.q ** b - 1) / 2) * CodedBKW.N(i, sigma_set, b, params.q)
                for i in range(1, t2 + 1)
            ]
        )
        C2 = float(C2_)
        for i in range(1, t2 + 1):
            C2 += float(
                ntop + ntest + sum([CodedBKW.N(j, sigma_set, b, params.q) for j in range(1, i + 1)])
            ) * (M + (i - 1) * (params.q ** b - 1) / 2)
        assert C2 >= 0

        # Equation (10)
        C3 = M * ntop * (2 * zeta + 1) ** ntop
        assert C3 >= 0

        # Equation (11)
        C4_ = 4 * M * ntest
        C4 = C4_ + (2 * zeta + 1) ** ntop * (
            cfft * params.q ** (ell + 1) * (ell + 1) * log(params.q, 2) + params.q ** (ell + 1)
        )
        assert C4 >= 0

        C = (C0 + C1 + C2 + C3 + C4) / (
            erf(zeta / sqrt(2 * params.Xe.stddev)) ** ntop
        )  # TODO don't ignore success probability
        cost["rop"] = float(C)
        cost["mem"] = (t1 + t2) * params.q ** b

        cost = cost.reorder("rop", "m", "mem", "b", "t1", "t2")
        cost["tag"] = "coded-bkw"
        cost["problem"] = params
        Logging.log("bkw", log_level + 1, f"{repr(cost)}")

        return cost

    @classmethod
    def b(
        cls,
        params: LWEParameters,
        ntest=None,
        log_level=1,
    ):
        def sf(x, best):
            return (x["rop"] <= best["rop"]) and (best["m"] > params.m or x["m"] <= params.m)

        # the outer search is over b, which determines the size of the tables: q^b
        b_max = 3 * ceil(log(params.q, 2))
        with local_minimum(2, b_max, smallerf=sf) as it_b:
            for b in it_b:
                # the inner search is over t2, the number of coded steps
                t2_max = min(params.n // b, ceil(3 * log(params.q, 2)))
                with local_minimum(2, t2_max, smallerf=sf) as it_t2:
                    for t2 in it_t2:
                        y = cls.cost(b=b, t2=t2, ntest=ntest, params=params)
                        it_t2.update(y)
                    it_b.update(it_t2.y)
            best = it_b.y

        # the search cannot fail. It just outputs some X with X["oracle"]>m.
        if best["m"] > params.m:
            raise InsufficientSamplesError(
                f"Got m≈2^{float(log(params.m, 2.0)):.1f} samples, but require ≈2^{float(log(best['m'],2.0)):.1f}.",
                best["m"],
            )
        return best

    def __call__(
        self,
        params: LWEParameters,
        ntest=None,
        log_level=1,
    ):
        """
        Coded-BKW as described in [C:GuoJohSta15]_.

        :param params: LWE parameters
        :param ntest: Number of coordinates to hypothesis test.
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``b``: BKW tables have size `q^b`.
        - ``t1``: Number of plain BKW tables.
        - ``t2``: Number of Coded-BKW tables.
        - ``ℓ``: Hypothesis testing has tables of size `q^{ℓ+1}`
        - ``#cod``: Number of coding steps.
        - ``#top``: Number of guessing steps (typically zero)
        - ``#test``: Number of coordinates to do hypothesis testing on.

        EXAMPLE::

            >>> from sage.all import oo
            >>> from estimator import *
            >>> LWE.coded_bkw(LightSaber.updated(m=oo))
            rop: ≈2^171.7, m: ≈2^159.4, mem: ≈2^160.4, b: 12, t1: 3, t2: 18, ℓ: 11, #cod: 423, #top: 1...

        We may need to amplify the number of samples, which modifies the noise distribution::

            >>> from sage.all import oo
            >>> from estimator import *
            >>> LightSaber
            LWEParameters(n=512, q=8192, Xs=D(σ=1.58), Xe=D(σ=2.00), m=512, tag='LightSaber')
            >>> cost = LWE.coded_bkw(LightSaber); cost
            rop: ≈2^184.3, m: ≈2^172.2, mem: ≈2^173.2, b: 13, t1: 0, t2: 18, ℓ: 12, #cod: 456, #top: 0...
            >>> cost["problem"]
            LWEParameters(n=512, q=8192, Xs=D(σ=1.58), Xe=D(σ=10.39), m=..., tag='LightSaber')

        .. note :: See also [C:KirFou15]_.

        """
        params = LWEParameters.normalize(params)
        try:
            cost = self.b(params, ntest=ntest, log_level=log_level)
        except InsufficientSamplesError as e:
            m = e.args[1]
            while True:
                params_ = params.amplify_m(m)
                try:
                    cost = self.b(params_, ntest=ntest, log_level=log_level)
                    break
                except InsufficientSamplesError as e:
                    m = e.args[1]

        return cost


coded_bkw = CodedBKW()
