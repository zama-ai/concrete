# -*- coding: utf-8 -*-
"""
Generic multiplicative composition of guessing some components of the LWE secret and some LWE solving algorithm.

By "multiplicative" we mean that costs multiply rather than add. It is often possible to achieve
some form of additive composition, i.e. this strategy is rarely the most efficient.

"""

from sage.all import log, floor, ceil, binomial
from sage.all import sqrt, pi, exp, RR, ZZ, oo, round, e

from .conf import mitm_opt
from .cost import Cost
from .errors import InsufficientSamplesError, OutOfBoundsError
from .lwe_parameters import LWEParameters
from .prob import amplify as prob_amplify
from .prob import drop as prob_drop
from .prob import amplify_sigma
from .util import local_minimum
from .nd import sigmaf


def log2(x):
    return log(x, 2)


class guess_composition:
    def __init__(self, f):
        """
        Create a generic composition of guessing and `f`.
        """
        self.f = f
        self.__name__ = f"{f.__name__}+guessing"

    @classmethod
    def dense_solve(cls, f, params, log_level=5, **kwds):
        """
        Guess components of a dense secret then call `f`.

        :param f: Some object consuming `params` and outputting some `cost`
        :param params: LWE parameters.

        """
        base = params.Xs.bounds[1] - params.Xs.bounds[0] + 1

        baseline_cost = f(params, **kwds)

        max_zeta = min(floor(log(baseline_cost["rop"], base)), params.n)

        with local_minimum(0, max_zeta, log_level=log_level) as it:
            for zeta in it:
                search_space = base ** zeta
                cost = f(params.updated(n=params.n - zeta), log_level=log_level + 1, **kwds)
                repeated_cost = cost.repeat(search_space)
                repeated_cost["zeta"] = zeta
                it.update(repeated_cost)
            return it.y

    @classmethod
    def gammaf(cls, n, h, zeta, base, g=lambda x: x):
        """
        Find optimal hamming weight for sparse guessing.

        Let `s` be a vector of dimension `n` where we expect `h` non-zero entries. We are ignoring `η-γ`
        components and are guessing `γ`. This succeeds with some probability given by ``prob_drop(n, h,
        ζ, γ)``. Exhaustively searching the guesses takes `binomial(n, γ) ⋅ b^γ` steps where `b` is the
        number of non-zero values in a component of `s`. We call a `γ` optimal if it minimizes the
        overall number of repetitions that need to be performed to succeed with probability 99%.

        :param n: vector dimension
        :param h: hamming weight of the vector
        :param zeta: number of ignored + guesses components
        :param base: number of possible non-zero scalars
        :param g: We do not consider search space directly by `g()` applied to it (think time-memory
                  trade-offs).
        :returns: (number of repetitions, γ, size of the search space, probability of success)

        """
        if not zeta:
            return 1, 0, 0, 1.0

        search_space = 0
        gamma = 0
        probability = 0
        best = None, None, None, None
        while gamma < min(h, zeta):
            probability += prob_drop(n, h, zeta, fail=gamma)
            search_space += binomial(zeta, gamma) * base ** gamma
            repeat = prob_amplify(0.99, probability) * g(search_space)
            if best[0] is None or repeat < best[0]:
                best = repeat, gamma, search_space, probability
                gamma += 1
            else:
                break
        return best

    @classmethod
    def sparse_solve(cls, f, params, log_level=5, **kwds):
        """
        Guess components of a sparse secret then call `f`.

        :param f: Some object consuming `params` and outputting some `cost`
        :param params: LWE parameters.
        """
        base = params.Xs.bounds[1] - params.Xs.bounds[0]  # we exclude zero
        h = ceil(len(params.Xs) * params.Xs.density)  # nr of non-zero entries

        with local_minimum(0, params.n - 40, log_level=log_level) as it:
            for zeta in it:
                single_cost = f(params.updated(n=params.n - zeta), log_level=log_level + 1, **kwds)
                repeat, gamma, search_space, probability = cls.gammaf(params.n, h, zeta, base)
                cost = single_cost.repeat(repeat)
                cost["zeta"] = zeta
                cost["|S|"] = search_space
                cost["prop"] = probability
                it.update(cost)
            return it.y

    def __call__(self, params, log_level=5, **kwds):
        """
        Guess components of a secret then call `f`.

        :param params: LWE parameters.

        EXAMPLE::

            >>> from estimator import *
            >>> from estimator.lwe_guess import guess_composition
            >>> guess_composition(LWE.primal_usvp)(Kyber512.updated(Xs=ND.SparseTernary(512, 16)))
            rop: ≈2^102.8, red: ≈2^102.8, δ: 1.008705, β: 113, d: 421, tag: usvp, ↻: ≈2^37.5, ζ: 265, |S|: 1, ...

        Compare::

            >>> LWE.primal_hybrid(Kyber512.updated(Xs=ND.SparseTernary(512, 16)))
            rop: ≈2^86.6, red: ≈2^85.7, svp: ≈2^85.6, β: 104, η: 2, ζ: 371, |S|: ≈2^91.1, d: 308, prob: ≈2^-21.3, ...

        """
        if params.Xs.is_sparse:
            return self.sparse_solve(self.f, params, log_level, **kwds)
        else:
            return self.dense_solve(self.f, params, log_level, **kwds)


class ExhaustiveSearch:
    def __call__(self, params: LWEParameters, success_probability=0.99, quantum: bool = False):
        """
        Estimate cost of solving LWE via exhaustive search.

        :param params: LWE parameters
        :param success_probability: the targeted success probability
        :param quantum: use estimate for quantum computer (we simply take the square root of the search space)
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: memory requirement in integers mod q.
        - ``m``: Required number of samples to distinguish the correct solution with high probability.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=64, q=2**40, Xs=ND.UniformMod(2), Xe=ND.DiscreteGaussian(3.2))
            >>> exhaustive_search(params)
            rop: ≈2^73.6, mem: ≈2^72.6, m: 397.198
            >>> params = LWE.Parameters(n=1024, q=2**40, Xs=ND.SparseTernary(n=1024, p=32), Xe=ND.DiscreteGaussian(3.2))
            >>> exhaustive_search(params)
            rop: ≈2^417.3, mem: ≈2^416.3, m: ≈2^11.2

        """
        params = LWEParameters.normalize(params)

        # there are two stages: enumeration and distinguishing, so we split up the success_probability
        probability = sqrt(success_probability)

        try:
            size = params.Xs.support_size(n=params.n, fraction=probability)
        except NotImplementedError:
            # not achieving required probability with search space
            # given our settings that means the search space is huge
            # so we approximate the cost with oo
            return Cost(rop=oo, mem=oo, m=1)

        if quantum:
            size = size.sqrt()

        # set m according to [ia.cr/2020/515]
        sigma = params.Xe.stddev / params.q
        m_required = RR(
            8 * exp(4 * pi * pi * sigma * sigma) * (log(size) - log(log(1 / probability)))
        )

        if params.m < m_required:
            raise InsufficientSamplesError(
                f"Exhaustive search: Need {m_required} samples but only {params.m} available."
            )
        else:
            m = m_required

        # we can compute A*s for all candidate s in time 2*size*m using
        # (the generalization [ia.cr/2021/152] of) the recursive algorithm
        # from [ia.cr/2020/515]
        cost = 2 * size * m

        ret = Cost(rop=cost, mem=cost / 2, m=m)
        return ret

    __name__ = "exhaustive_search"


exhaustive_search = ExhaustiveSearch()


class MITM:

    locality = 0.05

    def X_range(self, nd):
        if nd.is_bounded:
            a, b = nd.bounds
            return b - a + 1, 1.0
        else:
            # setting fraction=0 to ensure that support size does not
            # throw error. we'll take the probability into account later
            rng = nd.support_size(n=1, fraction=0.0)
            return rng, nd.gaussian_tail_prob

    def local_range(self, center):
        return ZZ(floor((1 - self.locality) * center)), ZZ(ceil((1 + self.locality) * center))

    def mitm_analytical(self, params: LWEParameters, success_probability=0.99):
        nd_rng, nd_p = self.X_range(params.Xe)
        delta = nd_rng / params.q  # possible error range scaled

        sd_rng, sd_p = self.X_range(params.Xs)

        # determine the number of elements in the tables depending on splitting dim
        n = params.n
        k = round(n / (2 - delta))
        # we could now call self.cost with this k, but using our model below seems
        # about 3x faster and reasonably accurate

        if params.Xs.is_sparse:
            h = params.Xs.get_hamming_weight(n=params.n)
            split_h = round(h * k / n)
            success_probability_ = (
                binomial(k, split_h) * binomial(n - k, h - split_h) / binomial(n, h)
            )

            logT = RR(h * (log2(n) - log2(h) + log2(sd_rng - 1) + log2(e))) / (2 - delta)
            logT -= RR(log2(h) / 2)
            logT -= RR(h * h * log2(e) / (2 * n * (2 - delta) ** 2))
        else:
            success_probability_ = 1.0
            logT = k * log(sd_rng, 2)

        m_ = max(1, round(logT + log(logT, 2)))
        if params.m < m_:
            raise InsufficientSamplesError(
                f"MITM: Need {m_} samples but only {params.m} available."
            )

        # since m = logT + loglogT and rop = T*m, we have rop=2^m
        ret = Cost(rop=RR(2 ** m_), mem=2 ** logT * m_, m=m_, k=ZZ(k))
        repeat = prob_amplify(success_probability, sd_p ** n * nd_p ** m_ * success_probability_)
        return ret.repeat(times=repeat)

    def cost(
        self,
        params: LWEParameters,
        k: int,
        success_probability=0.99,
    ):
        nd_rng, nd_p = self.X_range(params.Xe)
        delta = nd_rng / params.q  # possible error range scaled

        sd_rng, sd_p = self.X_range(params.Xs)
        n = params.n

        if params.Xs.is_sparse:
            h = params.Xs.get_hamming_weight(n=n)

            # we assume the hamming weight to be distributed evenly across the two parts
            # if not we can rerandomize on the coordinates and try again -> repeat
            split_h = round(h * k / n)
            size_tab = RR((sd_rng - 1) ** split_h * binomial(k, split_h))
            size_sea = RR((sd_rng - 1) ** (h - split_h) * binomial(n - k, h - split_h))
            success_probability_ = (
                binomial(k, split_h) * binomial(n - k, h - split_h) / binomial(n, h)
            )
        else:
            size_tab = sd_rng ** k
            size_sea = sd_rng ** (n - k)
            success_probability_ = 1

        # we set m such that it approximately minimizes the search cost per query as
        # a reasonable starting point and then optimize around it
        m_ = ceil(max(log2(size_tab) + log2(log2(size_tab)), 1))
        a, b = self.local_range(m_)
        with local_minimum(a, b, smallerf=lambda x, best: x[1] <= best[1]) as it:
            for m in it:
                # for search we effectively build a second table and for each entry, we expect
                # 2^( m * 4 * B / q) = 2^(delta * m) table look ups + a l_oo computation (costing m)
                # for every hit in the table (which has probability T/2^m)
                cost = (m, size_sea * (2 * m + 2 ** (delta * m) * (1 + size_tab * m / 2 ** m)))
                it.update(cost)
            m, cost_search = it.y
        m = min(m, params.m)

        # building the table costs 2*T*m using the generalization [ia.cr/2021/152] of
        # the recursive algorithm from [ia.cr/2020/515]
        cost_table = size_tab * 2 * m

        ret = Cost(rop=(cost_table + cost_search), m=m, k=k)
        ret["mem"] = size_tab * (k + m) + size_sea * (n - k + m)
        repeat = prob_amplify(success_probability, sd_p ** n * nd_p ** m * success_probability_)
        return ret.repeat(times=repeat)

    def __call__(self, params: LWEParameters, success_probability=0.99, optimization=mitm_opt):
        """
        Estimate cost of solving LWE via Meet-In-The-Middle attack.

        :param params: LWE parameters
        :param success_probability: the targeted success probability
        :param model: Either "analytical" (faster, default) or "numerical" (more accurate)
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: memory requirement in integers mod q.
        - ``m``: Required number of samples to distinguish the correct solution with high probability.
        - ``k``: Splitting dimension.
        - ``↻``: Repetitions required to achieve targeted success probability

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=64, q=2**40, Xs=ND.UniformMod(2), Xe=ND.DiscreteGaussian(3.2))
            >>> mitm(params)
            rop: ≈2^37.0, mem: ≈2^37.2, m: 37, k: 32, ↻: 1
            >>> mitm(params, optimization="numerical")
            rop: ≈2^39.2, m: 36, k: 32, mem: ≈2^39.1, ↻: 1
            >>> params = LWE.Parameters(n=1024, q=2**40, Xs=ND.SparseTernary(n=1024, p=32), Xe=ND.DiscreteGaussian(3.2))
            >>> mitm(params)
            rop: ≈2^215.4, mem: ≈2^210.2, m: ≈2^13.1, k: 512, ↻: 43
            >>> mitm(params, optimization="numerical")
            rop: ≈2^216.0, m: ≈2^13.1, k: 512, mem: ≈2^211.4, ↻: 43

        """
        Cost.register_impermanent(rop=True, mem=False, m=True, k=False)

        params = LWEParameters.normalize(params)

        nd_rng, _ = self.X_range(params.Xe)
        if nd_rng >= params.q:
            # MITM attacks cannot handle an error this large.
            return Cost(rop=oo, mem=oo, m=0, k=0)

        if "analytical" in optimization:
            return self.mitm_analytical(params=params, success_probability=success_probability)
        elif "numerical" in optimization:
            with local_minimum(1, params.n - 1) as it:
                for k in it:
                    cost = self.cost(k=k, params=params, success_probability=success_probability)
                    it.update(cost)
                ret = it.y
                # if the noise is large, the curve might not be convex, so the above minimum
                # is not correct. Interestingly, in these cases, it seems that k=1 might be smallest
                ret1 = self.cost(k=1, params=params, success_probability=success_probability)
                return min(ret, ret1)
        else:
            raise ValueError("Unknown optimization method for MITM.")

    __name__ = "mitm"


mitm = MITM()


class Distinguisher:
    def __call__(self, params: LWEParameters, success_probability=0.99):
        """
        Estimate cost of distinguishing a 0-dimensional LWE instance from uniformly random,
        which is essentially the number of samples required.

        :param params: LWE parameters
        :param success_probability: the targeted success probability
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: memory requirement in integers mod q.
        - ``m``: Required number of samples to distinguish.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=0, q=2 ** 32, Xs=ND.UniformMod(2), Xe=ND.DiscreteGaussian(2 ** 32))
            >>> distinguish(params)
            rop: ≈2^60.0, mem: ≈2^60.0, m: ≈2^60.0

        """

        if params.n > 0:
            raise OutOfBoundsError("Secret dimension should be 0 for distinguishing. Try exhaustive search for n > 0.")
        m = amplify_sigma(success_probability, sigmaf(params.Xe.stddev), params.q)
        if (m > params.m):
            raise InsufficientSamplesError("Not enough samples to distinguish with target advantage.")
        return Cost(rop=m, mem=m, m=m)

    __name__ = "distinguish"


distinguish = Distinguisher()
