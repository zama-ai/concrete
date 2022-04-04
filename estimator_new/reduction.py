# -*- coding: utf-8 -*-
"""
Cost estimates for lattice redution.
"""

from sage.all import ZZ, RR, pi, e, find_root, ceil, log, oo, round
from scipy.optimize import newton


def _delta(beta):
    """
    Compute δ from block size β without enforcing β ∈ ZZ.

    δ for β ≤ 40 were computed as follows:

    ```
    # -*- coding: utf-8 -*-
    from fpylll import BKZ, IntegerMatrix

    from multiprocessing import Pool
    from sage.all import mean, sqrt, exp, log, cputime

    d, trials = 320, 32

    def f((A, beta)):

        par = BKZ.Param(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT)
        q = A[-1, -1]
        d = A.nrows
        t = cputime()
        A = BKZ.reduction(A, par, float_type="dd")
        t = cputime(t)
        return t, exp(log(A[0].norm()/sqrt(q).n())/d)

    if __name__ == '__main__':
        for beta in (5, 10, 15, 20, 25, 28, 30, 35, 40):
            delta = []
            t = []
            i = 0
              while i < trials:
                threads = int(open("delta.nthreads").read()) # make sure this file exists
                pool = Pool(threads)
                A = [(IntegerMatrix.random(d, "qary", beta=d//2, bits=50), beta) for j in range(threads)]
                for (t_, delta_) in pool.imap_unordered(f, A):
                    t.append(t_)
                    delta.append(delta_)
                i += threads
                print u"β: %2d, δ_0: %.5f, time: %5.1fs, (%2d,%2d)"%(beta, mean(delta), mean(t), i, threads)
            print
    ```

    """
    small = (
        (2, 1.02190),  # noqa
        (5, 1.01862),  # noqa
        (10, 1.01616),
        (15, 1.01485),
        (20, 1.01420),
        (25, 1.01342),
        (28, 1.01331),
        (40, 1.01295),
    )

    if beta <= 2:
        return RR(1.0219)
    elif beta < 40:
        for i in range(1, len(small)):
            if small[i][0] > beta:
                return RR(small[i - 1][1])
    elif beta == 40:
        return RR(small[-1][1])
    else:
        return RR(beta / (2 * pi * e) * (pi * beta) ** (1 / beta)) ** (1 / (2 * (beta - 1)))


def delta(beta):
    """
    Compute root-Hermite factor δ from block size β.

    :param beta: Block size.
    """
    beta = ZZ(round(beta))
    return _delta(beta)


def _beta_secant(delta):
    """
    Estimate required block size β for a given root-Hermite factor δ based on [PhD:Chen13]_.

    :param delta: Root-Hermite factor.

    EXAMPLE::

        >>> import estimator.reduction as RC
        >>> 50 == RC._beta_secant(1.0121)
        True
        >>> 100 == RC._beta_secant(1.0093)
        True
        >>> RC._beta_secant(1.0024) # Chen reports 800
        808

    """
    # newton() will produce a "warning", if two subsequent function values are
    # indistinguishable (i.e. equal in terms of machine precision). In this case
    # newton() will return the value beta in the middle between the two values
    # k1,k2 for which the function values were indistinguishable.
    # Since f approaches zero for beta->+Infinity, this may be the case for very
    # large inputs, like beta=1e16.
    # For now, these warnings just get printed and the value beta is used anyways.
    # This seems reasonable, since for such large inputs the exact value of beta
    # doesn't make such a big difference.
    try:
        beta = newton(
            lambda beta: RR(_delta(beta) - delta),
            100,
            fprime=None,
            args=(),
            tol=1.48e-08,
            maxiter=500,
        )
        beta = ceil(beta)
        if beta < 40:
            # newton may output beta < 40. The old beta method wouldn't do this. For
            # consistency, call the old beta method, i.e. consider this try as "failed".
            raise RuntimeError("β < 40")
        return beta
    except (RuntimeError, TypeError):
        # if something fails, use old beta method
        beta = _beta_simple(delta)
        return beta


def _beta_find_root(delta):
    """
    Estimate required block size β for a given root-Hermite factor δ based on [PhD:Chen13]_.

    :param delta: Root-Hermite factor.

    TESTS::

        >>> import estimator.reduction as RC
        >>> RC._beta_find_root(RC.delta(500))
        500

    """
    # handle beta < 40 separately
    beta = ZZ(40)
    if _delta(beta) < delta:
        return beta

    try:
        beta = find_root(lambda beta: RR(_delta(beta) - delta), 40, 2 ** 16, maxiter=500)
        beta = ceil(beta - 1e-8)
    except RuntimeError:
        # finding root failed; reasons:
        # 1. maxiter not sufficient
        # 2. no root in given interval
        beta = _beta_simple(delta)
    return beta


def _beta_simple(delta):
    """
    Estimate required block size β for a given root-Hermite factor δ based on [PhD:Chen13]_.

    :param delta: Root-Hermite factor.

    TESTS::

        >>> import estimator.reduction as RC
        >>> RC._beta_simple(RC.delta(500))
        501

    """
    beta = ZZ(40)

    while _delta(2 * beta) > delta:
        beta *= 2
    while _delta(beta + 10) > delta:
        beta += 10
    while True:
        if _delta(beta) < delta:
            break
        beta += 1

    return beta


def beta(delta):
    """
    Estimate required block size β for a given root-hermite factor δ based on [PhD:Chen13]_.

    :param delta: Root-hermite factor.

    EXAMPLE::

        >>> import estimator.reduction as RC
        >>> 50 == RC.beta(1.0121)
        True
        >>> 100 == RC.beta(1.0093)
        True
        >>> RC.beta(1.0024) # Chen reports 800
        808

    """
    # TODO: decide for one strategy (secant, find_root, old) and its error handling
    beta = _beta_find_root(delta)
    return beta


# BKZ Estimates


def svp_repeat(beta, d):
    """
    Return number of SVP calls in BKZ-β.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.

    .. note :: Loosely based on experiments in [PhD:Chen13].

    .. note :: When d ≤ β we return 1.

    """
    if beta < d:
        return 8 * d
    else:
        return 1


def LLL(d, B=None):
    """
    Runtime estimation for LLL algorithm based on [AC:CheNgu11]_.

    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    """
    if B:
        return d ** 3 * B ** 2
    else:
        return d ** 3  # ignoring B for backward compatibility


def _BDGL16_small(beta, d, B=None):
    """
    Runtime estimation given β and assuming sieving is used to realise the SVP oracle for small
    dimensions following [SODA:BDGL16]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    TESTS::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC._BDGL16_small(500, 1024), 2.0)
        222.9

    """
    return LLL(d, B) + ZZ(2) ** RR(0.387 * beta + 16.4 + log(svp_repeat(beta, d), 2))


def _BDGL16_asymptotic(beta, d, B=None):
    """
    Runtime estimation given `β` and assuming sieving is used to realise the SVP oracle following [SODA:BDGL16]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    TESTS::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC._BDGL16_asymptotic(500, 1024), 2.0)
        175.4
    """
    # TODO we simply pick the same additive constant 16.4 as for the experimental result in [SODA:BDGL16]_
    return LLL(d, B) + ZZ(2) ** RR(0.292 * beta + 16.4 + log(svp_repeat(beta, d), 2))


def BDGL16(beta, d, B=None):
    """
    Runtime estimation given `β` and assuming sieving is used to realise the SVP oracle following [SODA:BDGL16]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.BDGL16(500, 1024), 2.0)
        175.4

    """
    # TODO this is somewhat arbitrary
    if beta <= 90:
        return _BDGL16_small(beta, d, B)
    else:
        return _BDGL16_asymptotic(beta, d, B)


def LaaMosPol14(beta, d, B=None):
    """
    Runtime estimation for quantum sieving following [EPRINT:LaaMosPol14]_ and [PhD:Laarhoven15]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.LaaMosPol14(500, 1024), 2.0)
        161.9

    """
    return LLL(d, B) + ZZ(2) ** RR((0.265 * beta + 16.4 + log(svp_repeat(beta, d), 2)))


def CheNgu12(beta, d, B=None):
    """
    Runtime estimation given β and assuming [CheNgu12]_ estimates are correct.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    The constants in this function were derived as follows based on Table 4 in
    [CheNgu12]_::

        >>> from sage.all import var, find_fit
        >>> dim = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
        >>> nodes = [39.0, 44.0, 49.0, 54.0, 60.0, 66.0, 72.0, 78.0, 84.0, 96.0, 99.0, 105.0, 111.0, 120.0, 127.0, 134.0]  # noqa
        >>> times = [c + log(200,2).n() for c in nodes]
        >>> T = list(zip(dim, nodes))
        >>> var("a,b,c,beta")
        (a, b, c, beta)
        >>> f = a*beta*log(beta, 2.0) + b*beta + c
        >>> f = f.function(beta)
        >>> f.subs(find_fit(T, f, solution_dict=True))
        beta |--> 0.2701...*beta*log(beta) - 1.0192...*beta + 16.10...

    The estimation 2^(0.18728 β⋅log_2(β) - 1.019⋅β + 16.10) is of the number of enumeration
    nodes, hence we need to multiply by the number of cycles to process one node. This cost per
    node is typically estimated as 64.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.CheNgu12(500, 1024), 2.0)
        365.70...

    """
    repeat = svp_repeat(beta, d)
    cost = RR(
        0.270188776350190 * beta * log(beta)
        - 1.0192050451318417 * beta
        + 16.10253135200765
        + log(100, 2)
    )
    return LLL(d, B) + repeat * ZZ(2) ** cost


def ABFKSW20(beta, d, B=None):
    """
    Enumeration cost according to [C:ABFKSW20]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.ABFKSW20(500, 1024), 2.0)
        316.26...

    """
    if 1.5 * beta >= d or beta <= 92:  # 1.5β is a bit arbitrary, β≤92 is the crossover point
        cost = RR(0.1839 * beta * log(beta, 2) - 0.995 * beta + 16.25 + log(64, 2))
    else:
        cost = RR(0.125 * beta * log(beta, 2) - 0.547 * beta + 10.4 + log(64, 2))

    repeat = svp_repeat(beta, d)

    return LLL(d, B) + repeat * ZZ(2) ** cost


def ABLR21(beta, d, B=None):
    """
    Enumeration cost according to [C:ABLR21]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.ABLR21(500, 1024), 2.0)
        278.20...

    """
    if 1.5 * beta >= d or beta <= 97:  # 1.5β is a bit arbitrary, 97 is the crossover
        cost = RR(0.1839 * beta * log(beta, 2) - 1.077 * beta + 29.12 + log(64, 2))
    else:
        cost = RR(0.1250 * beta * log(beta, 2) - 0.654 * beta + 25.84 + log(64, 2))

    repeat = svp_repeat(beta, d)

    return LLL(d, B) + repeat * ZZ(2) ** cost


def ADPS16(beta, d, B=None, mode="classical"):
    """
    Runtime estimation from [USENIX:ADPS16]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.ADPS16(500, 1024), 2.0)
        146.0
        >>> log(RC.ADPS16(500, 1024, mode="quantum"), 2.0)
        132.5
        >>> log(RC.ADPS16(500, 1024, mode="paranoid"), 2.0)
        103.75

    """

    if mode not in ("classical", "quantum", "paranoid"):
        raise ValueError(f"Mode {mode} not understood.")

    c = {
        "classical": 0.2920,
        "quantum": 0.2650,  # paper writes 0.262 but this isn't right, see above
        "paranoid": 0.2075,
    }

    c = c[mode]

    return ZZ(2) ** RR(c * beta)


def d4f(beta):
    """
    Dimensions "for free" following [EC:Ducas18]_.

    :param beta: Block size ≥ 2.

    If β' is output by this function then sieving is expected to be required up to dimension β-β'.

    EXAMPLE::

        >>> import estimator.reduction as RC
        >>> RC.d4f(500)
        42.597...

    """
    return max(float(beta * log(4 / 3.0) / log(beta / (2 * pi * e))), 0.0)


# These are not asymptotic expressions but compress the data in [AC:AGPS20]_ which covers up to
# β = 1024
NN_AGPS = {
    "all_pairs-classical": {"a": 0.4215069316613415, "b": 20.1669683097337},
    "all_pairs-dw": {"a": 0.3171724396445732, "b": 25.29828951733785},
    "all_pairs-g": {"a": 0.3155285835002801, "b": 22.478746811528048},
    "all_pairs-ge19": {"a": 0.3222895263943544, "b": 36.11746438609666},
    "all_pairs-naive_classical": {"a": 0.4186251294633655, "b": 9.899382654377058},
    "all_pairs-naive_quantum": {"a": 0.31401512556555794, "b": 7.694659515948326},
    "all_pairs-t_count": {"a": 0.31553282515234704, "b": 20.878594142502994},
    "list_decoding-classical": {"a": 0.2988026130564745, "b": 26.011121212891872},
    "list_decoding-dw": {"a": 0.26944796385592995, "b": 28.97237346443934},
    "list_decoding-g": {"a": 0.26937450988892553, "b": 26.925140365395972},
    "list_decoding-ge19": {"a": 0.2695210400018704, "b": 35.47132142280775},
    "list_decoding-naive_classical": {"a": 0.2973130399197453, "b": 21.142124058689426},
    "list_decoding-naive_quantum": {"a": 0.2674316807758961, "b": 18.720680589028465},
    "list_decoding-t_count": {"a": 0.26945736714156543, "b": 25.913746774011887},
    "random_buckets-classical": {"a": 0.35586144233444716, "b": 23.082527816636638},
    "random_buckets-dw": {"a": 0.30704199612690264, "b": 25.581968903639485},
    "random_buckets-g": {"a": 0.30610964725102385, "b": 22.928235564044563},
    "random_buckets-ge19": {"a": 0.31089687599538407, "b": 36.02129978813208},
    "random_buckets-naive_classical": {"a": 0.35448283789554513, "b": 15.28878540793908},
    "random_buckets-naive_quantum": {"a": 0.30211421791887644, "b": 11.151745013027089},
    "random_buckets-t_count": {"a": 0.30614770082829745, "b": 21.41830142853265},
}


def Kyber(beta, d, B=None, nn="classical", C=5.46):
    """
    Runtime estimation from [Kyber20]_ and [AC:AGPS20]_.

    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.
    :param nn: Nearest neighbor cost model. We default to "ListDecoding" (i.e. BDGL16) and to
               the "depth × width" metric. Kyber uses "AllPairs".
    :param C: Progressive overhead lim_{β → ∞} ∑_{i ≤ β} 2^{0.292 i + o(i)}/2^{0.292 β + o(β)}.

    EXAMPLE::

        >>> from math import log
        >>> import estimator.reduction as RC
        >>> log(RC.Kyber(500, 1024), 2.0)
        174.16...
        >>> log(RC.Kyber(500, 1024, nn="list_decoding-ge19"), 2.0)
        170.23...

    """

    if beta < 20:  # goes haywire
        return CheNgu12(beta, d, B)

    if nn == "classical":
        nn = "list_decoding-classical"
    elif nn == "quantum":
        nn = "list_decoding-dw"

    svp_calls = C * max(d - beta, 1)
    # we do not round to the nearest integer to ensure cost is continuously increasing with β which
    # rounding can violate.
    beta_ = beta - d4f(beta)
    gate_count = 2 ** (NN_AGPS[nn]["a"] * beta_ + NN_AGPS[nn]["b"])
    return LLL(d, B=B) + svp_calls * gate_count


def cost(cost_model, beta, d, B=None, predicate=None, **kwds):
    """
    Return cost dictionary for computing vector of norm` δ_0^{d-1} Vol(Λ)^{1/d}` using provided lattice
    reduction algorithm.

    :param cost_model:
    :param beta: Block size ≥ 2.
    :param d: Lattice dimension.
    :param B: Bit-size of entries.
    :param predicate: if ``False`` cost will be infinity.

    EXAMPLE::

        >>> import estimator.reduction as RC
        >>> RC.cost(RC.ABLR21, 120, 500)
        rop: ≈2^68.9, red: ≈2^68.9, δ: 1.008435, β: 120, d: 500
        >>> RC.cost(RC.ABLR21, 120, 500, predicate=False)
        rop: ≈2^inf, red: ≈2^inf, δ: 1.008435, β: 120, d: 500

    """
    from .cost import Cost

    cost = cost_model(beta, d, B)
    delta_ = delta(beta)
    cost = Cost(rop=cost, red=cost, delta=delta_, beta=beta, d=d, **kwds)
    cost.register_impermanent(rop=True, red=True, delta=False, beta=False, d=False)
    if predicate is not None and not predicate:
        cost["red"] = oo
        cost["rop"] = oo
    return cost
