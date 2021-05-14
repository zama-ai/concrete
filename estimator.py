# -*- coding: utf-8 -*-
"""
Cost estimates for solving LWE.

.. moduleauthor:: Martin R. Albrecht <martinralbrecht@googlemail.com>

# Supported Secret Distributions #

The following distributions for the secret are supported:

- ``"normal"`` : normal form instances, i.e. the secret follows the noise distribution (alias: ``True``)
- ``alpha``: where α a positive real number, (discrete) Gaussian distribution with parameter α
- ``"uniform"`` : uniform mod q (alias: ``False``)
- ``(a,b)`` : uniform in the interval ``[a,…,b]``
- ``((a,b), h)`` : exactly ``h`` components are ``∈ [a,…,b]∖\\{0\\}``, all other components are zero

"""


# Imports
from collections import OrderedDict
from functools import partial
from sage.arith.srange import srange
from sage.calculus.var import var
from sage.functions.log import exp, log
from sage.functions.other import ceil, sqrt, floor, binomial
from sage.all import erf
from sage.interfaces.magma import magma
from sage.misc.all import cached_function, round, prod
from sage.numerical.optimize import find_root
from sage.rings.all import QQ, RR, ZZ, RealField, PowerSeriesRing, RDF
from sage.rings.infinity import PlusInfinity
from sage.structure.element import parent
from sage.symbolic.all import pi, e
from scipy.optimize import newton
import logging
import sage.crypto.lwe

oo = PlusInfinity()


class Logging:
    """
    Control level of detail being printed.
    """

    plain_logger = logging.StreamHandler()
    plain_logger.setFormatter(logging.Formatter("%(message)s"))

    detail_logger = logging.StreamHandler()
    detail_logger.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))

    logging.getLogger("estimator").handlers = [plain_logger]
    logging.getLogger("estimator").setLevel(logging.INFO)

    loggers = ("binsearch", "repeat", "guess")

    for logger in loggers:
        logging.getLogger(logger).handlers = [detail_logger]
        logging.getLogger(logger).setLevel(logging.INFO)

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    @staticmethod
    def set_level(lvl, loggers=None):
        """Set logging level

        :param lvl: one of `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`, `NOTSET`
        :param loggers: one of `Logging.loggers`, if `None` all loggers are used.

        """
        if loggers is None:
            loggers = Logging.loggers

        for logger in loggers:
            logging.getLogger(logger).setLevel(lvl)


# Utility Classes #


class OutOfBoundsError(ValueError):
    """
    Used to indicate a wrong value, for example delta_0 < 1.
    """

    pass


class InsufficientSamplesError(ValueError):
    """
    Used to indicate the number of samples given is too small.
    """

    pass


# Binary Search #


def binary_search_minimum(f, start, stop, param, extract=lambda x: x, *arg, **kwds):
    """
    Return minimum of `f` if `f` is convex.

    :param start: start of range to search
    :param stop:  stop of range to search (exclusive)
    :param param: the parameter to modify when calling `f`
    :param extract: comparison is performed on ``extract(f(param=?, *args, **kwds))``

    """
    return binary_search(f, start, stop, param, predictate=lambda x, best: extract(x) <= extract(best), *arg, **kwds)


def binary_search(f, start, stop, param, predicate=lambda x, best: x <= best, *arg, **kwds):
    """
    Searches for the best value in the interval [start,stop] depending on the given predicate.

    :param start: start of range to search
    :param stop: stop of range to search (exclusive)
    :param param: the parameter to modify when calling `f`
    :param predicate: comparison is performed by evaluating ``predicate(current, best)``
    """
    kwds[param] = stop
    D = {}
    D[stop] = f(*arg, **kwds)
    best = D[stop]
    b = ceil((start + stop) / 2)
    direction = 0
    while True:
        if b not in D:
            kwds[param] = b
            D[b] = f(*arg, **kwds)
        if b == start:
            best = D[start]
            break
        if not predicate(D[b], best):
            if direction == 0:
                start = b
                b = ceil((stop + b) / 2)
            else:
                stop = b
                b = floor((start + b) / 2)
        else:
            best = D[b]
            logging.getLogger("binsearch").debug(u"%4d, %s" % (b, best))
            if b - 1 not in D:
                kwds[param] = b - 1
                D[b - 1] = f(*arg, **kwds)
            if predicate(D[b - 1], best):
                stop = b
                b = floor((b + start) / 2)
                direction = 0
            else:
                if b + 1 not in D:
                    kwds[param] = b + 1
                    D[b + 1] = f(*arg, **kwds)
                if not predicate(D[b + 1], best):
                    break
                else:
                    start = b
                    b = ceil((stop + b) / 2)
                    direction = 1
    return best


class Param:
    """
    Namespace for processing LWE parameter sets.
    """

    @staticmethod
    def Regev(n, m=None, dict=False):
        """
        :param n: LWE dimension `n > 0`
        :param m: number of LWE samples `m > 0`
        :param dict: return a dictionary

        """
        if dict:
            return Param.dict(sage.crypto.lwe.Regev(n=n, m=m))
        else:
            return Param.tuple(sage.crypto.lwe.Regev(n=n, m=m))

    @staticmethod
    def LindnerPeikert(n, m=None, dict=False):
        """
        :param n: LWE dimension `n > 0`
        :param m: number of LWE samples `m > 0`
        :param dict: return a dictionary

        """
        if dict:
            return Param.dict(sage.crypto.lwe.LindnerPeikert(n=n, m=m))
        else:
            return Param.tuple(sage.crypto.lwe.LindnerPeikert(n=n, m=m))

    @staticmethod
    def tuple(lwe):
        """
        Return (n, α, q) given an LWE instance object.

        :param lwe: LWE object
        :returns: (n, α, q)

        """
        n = lwe.n
        q = lwe.K.order()
        try:
            alpha = alphaf(sigmaf(lwe.D.sigma), q)
        except AttributeError:
            # older versions of Sage use stddev, not sigma
            alpha = alphaf(sigmaf(lwe.D.stddev), q)
        return n, alpha, q

    @staticmethod
    def dict(lwe):
        """
        Return dictionary consisting of n, α, q and samples given an LWE instance object.

        :param lwe: LWE object
        :returns: "n": n, "alpha": α, "q": q, "samples": samples
        :rtype: dictionary

        """
        n, alpha, q = Param.tuple(lwe)
        m = lwe.m if lwe.m else oo
        return {"n": n, "alpha": alpha, "q": q, "m": m}

    @staticmethod
    def preprocess(n, alpha, q, success_probability=None, prec=None, m=oo):
        """
        Check if parameters n, α, q are sound and return correct types.
        Also, if given, the soundness of the success probability and the
        number of samples is ensured.
        """
        if n < 1:
            raise ValueError("LWE dimension must be greater than 0.")
        if alpha <= 0:
            raise ValueError("Fraction of noise must be > 0.")
        if q < 1:
            raise ValueError("LWE modulus must be greater than 0.")
        if m is not None and m < 1:
            raise InsufficientSamplesError(u"m=%d < 1" % m)

        if prec is None:
            try:
                prec = alpha.prec()
            except AttributeError:
                pass
            try:
                prec = max(success_probability.prec(), prec if prec else 0)
            except AttributeError:
                pass

            if prec is None or prec < 128:
                prec = 128
        RR = RealField(prec)
        n, alpha, q = ZZ(n), RR(alpha), ZZ(q)

        if m is not oo:
            m = ZZ(m)

        if success_probability is not None:
            if success_probability >= 1 or success_probability <= 0:
                raise ValueError("success_probability must be between 0 and 1.")
            return n, alpha, q, RR(success_probability)
        else:
            return n, alpha, q


# Error Parameter Conversions


def stddevf(sigma):
    """
    Gaussian width parameter σ → standard deviation

    :param sigma: Gaussian width parameter σ

    EXAMPLE::

        sage: from estimator import stddevf
        sage: stddevf(64.0)
        25.532...

        sage: stddevf(64)
        25.532...

        sage: stddevf(RealField(256)(64)).prec()
        256

    """

    try:
        prec = parent(sigma).prec()
    except AttributeError:
        prec = 0
    if prec > 0:
        FF = parent(sigma)
    else:
        FF = RR
    return FF(sigma) / FF(sqrt(2 * pi))


def sigmaf(stddev):
    """
    standard deviation → Gaussian width parameter σ

    :param sigma: standard deviation

    EXAMPLE::

        sage: from estimator import stddevf, sigmaf
        sage: n = 64.0
        sage: sigmaf(stddevf(n))
        64.000...

        sage: sigmaf(RealField(128)(1.0))
        2.5066282746310005024157652848110452530
        sage: sigmaf(1.0)
        2.50662827463100
        sage: sigmaf(1)
        2.50662827463100
        sage: sigmaf(1r)
        2.50662827463100

    """
    RR = parent(stddev)
    #  check that we got ourselves a real number type
    try:
        if (abs(RR(0.5) - 0.5) > 0.001):
            RR = RealField(53)  # hardcode something
    except TypeError:
        RR = RealField(53)  # hardcode something
    return RR(sqrt(2 * pi)) * stddev


def alphaf(sigma, q, sigma_is_stddev=False):
    """
    Gaussian width σ, modulus q → noise rate α

    :param sigma: Gaussian width parameter (or standard deviation if ``sigma_is_stddev`` is set)
    :param q: modulus `0 < q`
    :param sigma_is_stddev: if set then `sigma` is interpreted as the standard deviation

    :returns: α = σ/q or σ·sqrt(2π)/q depending on `sigma_is_stddev`

    """
    if sigma_is_stddev is False:
        return RR(sigma / q)
    else:
        return RR(sigmaf(sigma) / q)


class Cost:
    """
    Algorithms costs.
    """

    def __init__(self, data=None, **kwds):
        """

        :param data: we call ``OrderedDict(data)``

        """
        if data is None:
            self.data = OrderedDict()
        else:
            self.data = OrderedDict(data)

        for k, v in kwds.items():
            self.data[k] = v

    def str(self, keyword_width=None, newline=None, round_bound=2048, compact=False, unicode=True):
        """

        :param keyword_width:  keys are printed with this width
        :param newline:        insert a newline
        :param round_bound:    values beyond this bound are represented as powers of two
        :param compact:        do not add extra whitespace to align entries
        :param unicode:        use unicode to shorten representation

        EXAMPLE::

            sage: from estimator import Cost
            sage: s = Cost({"delta_0":5, "bar":2})
            sage: print(s)
            delta_0: 5, bar: 2

            sage: s = Cost([(u"delta_0", 5), ("bar",2)])
            sage: print(s)
            delta_0: 5, bar: 2

        """
        if unicode:
            unicode_replacements = {"delta_0": u"δ_0", "beta": u"β", "epsilon": u"ε"}
        else:
            unicode_replacements = {}

        format_strings = {
            u"beta": u"%s: %4d",
            u"d": u"%s: %4d",
            "b": "%s: %3d",
            "t1": "%s: %3d",
            "t2": "%s: %3d",
            "l": "%s: %3d",
            "ncod": "%s: %3d",
            "ntop": "%s: %3d",
            "ntest": "%s: %3d",
        }

        d = self.data
        s = []
        for k in d:
            v = d[k]
            kk = unicode_replacements.get(k, k)
            if keyword_width:
                fmt = u"%%%ds" % keyword_width
                kk = fmt % kk
            if not newline and k in format_strings:
                s.append(format_strings[k] % (kk, v))
            elif ZZ(1) / round_bound < v < round_bound or v == 0 or ZZ(-1) / round_bound > v > -round_bound:
                try:
                    if compact:
                        s.append(u"%s: %d" % (kk, ZZ(v)))
                    else:
                        s.append(u"%s: %8d" % (kk, ZZ(v)))
                except TypeError:
                    if v < 2.0 and v >= 0.0:
                        if compact:
                            s.append(u"%s: %.6f" % (kk, v))
                        else:
                            s.append(u"%s: %8.6f" % (kk, v))
                    else:
                        if compact:
                            s.append(u"%s: %.3f" % (kk, v))
                        else:
                            s.append(u"%s: %8.3f" % (kk, v))
            else:
                t = u"%s" % (u"≈" if unicode else "") + u"%s2^%.1f" % ("-" if v < 0 else "", log(abs(v), 2).n())
                if compact:
                    s.append(u"%s: %s" % (kk, t))
                else:
                    s.append(u"%s: %8s" % (kk, t))
        if not newline:
            if compact:
                return u", ".join(s)
            else:
                return u",  ".join(s)
        else:
            return u"\n".join(s)

    def reorder(self, first):
        """
        Return a new ordered dict from the key:value pairs in dictinonary but reordered such that the
        ``first`` keys come first.

        :param dictionary: input dictionary
        :param first: keys which should come first (in order)

        EXAMPLE::

            sage: from estimator import Cost
            sage: d = Cost([("a",1),("b",2),("c",3)]); d
            a:        1
            b:        2
            c:        3

            sage: d.reorder( ["b","c","a"])
            b:        2
            c:        3
            a:        1
        """
        keys = list(self.data)
        for key in first:
            keys.pop(keys.index(key))
        keys = list(first) + keys
        r = OrderedDict()
        for key in keys:
            r[key] = self.data[key]
        return Cost(r)

    def filter(self, keys):
        """
        Return new ordered dictinonary from dictionary restricted to the keys.

        :param dictionary: input dictionary
        :param keys: keys which should be copied (ordered)
        """
        r = OrderedDict()
        for key in keys:
            r[key] = self.data[key]
        return Cost(r)

    def repeat(self, times, select=None, lll=None):
        u"""
        Return a report with all costs multiplied by `times`.

        :param d:      a cost estimate
        :param times:  the number of times it should be run
        :param select: toggle which fields ought to be repeated and which shouldn't
        :param lll:    if set amplify lattice reduction times assuming the LLL algorithm suffices and costs ``lll``
        :returns:      a new cost estimate

        We maintain a local dictionary which decides if an entry is multiplied by `times` or not.
        For example, δ would not be multiplied but "#bop" would be. This check is strict such that
        unknown entries raise an error. This is to enforce a decision on whether an entry should be
        multiplied by `times` if the function `report` reports on is called `times` often.

        EXAMPLE::

            sage: from estimator import Param, dual
            sage: n, alpha, q = Param.Regev(128)

            sage: dual(n, alpha, q).repeat(2^10)
                rop:   2^91.1
                  m:   2^18.6
                red:   2^91.1
            delta_0: 1.008631
               beta:      115
                  d:      380
                |v|:  688.951
             repeat:   2^27.0
            epsilon: 0.007812

            sage: dual(n, alpha, q).repeat(1)
                rop:   2^81.1
                  m:      380
                red:   2^81.1
            delta_0: 1.008631
               beta:      115
                  d:      380
                |v|:  688.951
             repeat:   2^17.0
            epsilon: 0.007812

        """
        # TODO review this list
        do_repeat = {
            u"rop": True,
            u"red": True,
            u"babai": True,
            u"babai_op": True,
            u"epsilon": False,
            u"mem": False,
            u"delta_0": False,
            u"beta": False,
            u"k": False,
            u"D_reg": False,
            u"t": False,
            u"m": True,
            u"d": False,
            u"|v|": False,
            u"amplify": False,
            u"repeat": False,  # we deal with it below
            u"c": False,
            u"b": False,
            u"t1": False,
            u"t2": False,
            u"l": False,
            u"ncod": False,
            u"ntop": False,
            u"ntest": False,
        }

        if lll and self["red"] != self["rop"]:
            raise ValueError("Amplification via LLL was requested but 'red' != 'rop'")

        if select is not None:
            for key in select:
                do_repeat[key] = select[key]

        ret = OrderedDict()
        for key in self.data:
            try:
                if do_repeat[key]:
                    if lll and key in ("red", "rop"):
                        ret[key] = self[key] + times * lll
                    else:
                        ret[key] = times * self[key]
                else:
                    ret[key] = self.data[key]
            except KeyError:
                raise NotImplementedError(u"You found a bug, this function does not know about '%s' but should." % key)
        ret[u"repeat"] = times * ret.get("repeat", 1)
        return Cost(ret)

    def __rmul__(self, times):
        return self.repeat(times)

    def combine(self, right, base=None):
        """Combine ``left`` and ``right``.

        :param left: cost dictionary
        :param right: cost dictionary
        :param base: add entries to ``base``

        """
        if base is None:
            cost = Cost()
        else:
            cost = base
        for key in self.data:
            cost[key] = self.data[key]
        for key in right:
            cost[key] = right.data[key]
        return Cost(cost)

    def __add__(self, other):
        return self.combine(self, other)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def values(self):
        return self.data.values()

    def __str__(self):
        return self.str(unicode=False, compact=True)

    def __repr__(self):
        return self.str(unicode=False, newline=True, keyword_width=12)

    def __unicode__(self):
        return self.str(unicode=True)


class SDis:
    """
    Distributions of Secrets.
    """

    @staticmethod
    def is_sparse(secret_distribution):
        """Return true if the secret distribution is sparse

        :param secret_distribution: distribution of secret, see module level documentation for details

        EXAMPLES::

            sage: from estimator import SDis
            sage: SDis.is_sparse(True)
            False

            sage: SDis.is_sparse(0.1)
            False

            sage: SDis.is_sparse(0.2)
            False

            sage: SDis.is_sparse(((-1, 1), 64))
            True

            sage: SDis.is_sparse(((-3, 3), 64))
            True

            sage: SDis.is_sparse((-3, 3))
            False

        """
        try:
            (a, b), h = secret_distribution
            # TODO we could check h against n but then this function would have to depend on n
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def is_small(secret_distribution, alpha=None):
        """Return true if the secret distribution is small

        :param secret_distribution: distribution of secret, see module level documentation for details
        :param alpha: optional, used for comparing with `secret_distribution`

        EXAMPLES::

            sage: from estimator import SDis
            sage: SDis.is_small(False)
            False

            sage: SDis.is_small(True)
            True

            sage: SDis.is_small("normal")
            True

            sage: SDis.is_small(1)
            True

            sage: SDis.is_small(1.2)
            True

            sage: SDis.is_small(((-1, 1), 64))
            True

            sage: SDis.is_small(((-3, 3), 64))
            True

            sage: SDis.is_small((-3, 3))
            True

            sage: SDis.is_small((0, 1))
            True

        """

        if secret_distribution == "normal" or secret_distribution is True:
            return True
        try:
            if float(secret_distribution) > 0:
                if alpha is not None:
                    if secret_distribution <= alpha:
                        return True
                    else:
                        raise NotImplementedError("secret size %f > error size %f"%(secret_distribution, alpha))
                return True
        except TypeError:
            pass
        try:
            (a, b), h = secret_distribution
            return True
        except (TypeError, ValueError):
            try:
                (a, b) = secret_distribution
                return True
            except (TypeError, ValueError):
                return False

    @staticmethod
    def bounds(secret_distribution):
        """Return bounds of secret distribution

        :param secret_distribution: distribution of secret, see module level documentation for details

        EXAMPLES::

            sage: from estimator import SDis
            sage: SDis.bounds(False)
            Traceback (most recent call last):
            ...
            ValueError: Cannot extract bounds for secret.

            sage: SDis.bounds(True)
            Traceback (most recent call last):
            ...
            ValueError: Cannot extract bounds for secret.

            sage: SDis.bounds(0.1)
            Traceback (most recent call last):
            ...
            ValueError: Cannot extract bounds for secret.

            sage: SDis.bounds(0.12)
            Traceback (most recent call last):
            ...
            ValueError: Cannot extract bounds for secret.

            sage: SDis.bounds(((-1, 1), 64))
            (-1, 1)

            sage: SDis.bounds(((-3, 3), 64))
            (-3, 3)

            sage: SDis.bounds((-3, 3))
            (-3, 3)

        """
        try:
            (a, b) = secret_distribution
            try:
                (a, b), _ = (a, b)  # noqa
            except (TypeError, ValueError):
                pass
            return a, b
        except (TypeError, ValueError):
            raise ValueError("Cannot extract bounds for secret.")

    @staticmethod
    def is_bounded_uniform(secret_distribution):
        """Return true if the secret is bounded uniform (sparse or not).

        :param secret_distribution: distribution of secret, see module level documentation for
            details

        EXAMPLES::

            sage: from estimator import SDis
            sage: SDis.is_bounded_uniform(False)
            False

            sage: SDis.is_bounded_uniform(True)
            False

            sage: SDis.is_bounded_uniform(0.1)
            False

            sage: SDis.is_bounded_uniform(0.12)
            False

            sage: SDis.is_bounded_uniform(((-1, 1), 64))
            True

            sage: SDis.is_bounded_uniform(((-3, 3), 64))
            True

            sage: SDis.is_bounded_uniform((-3, 3))
            True

        ..  note :: This function requires the bounds to be of opposite sign, as scaling code does
            not handle the other case.

        """

        try:
            # next will fail if not bounded_uniform
            a, b = SDis.bounds(secret_distribution)

            # check bounds are around 0, otherwise not implemented
            if a <= 0 and 0 <= b:
                return True
        except (TypeError, ValueError):
            pass

        return False

    @staticmethod
    def is_ternary(secret_distribution):
        """Return true if the secret is ternary (sparse or not)

        :param secret_distribution: distribution of secret, see module level documentation for details

        EXAMPLES::

            sage: from estimator import SDis
            sage: SDis.is_ternary(False)
            False

            sage: SDis.is_ternary(True)
            False

            sage: SDis.is_ternary(0.1)
            False

            sage: SDis.is_ternary(0.12)
            False

            sage: SDis.is_ternary(((-1, 1), 64))
            True

            sage: SDis.is_ternary((-1, 1))
            True

            sage: SDis.is_ternary(((-3, 3), 64))
            False

            sage: SDis.is_ternary((-3, 3))
            False

        """

        if SDis.is_bounded_uniform(secret_distribution):
            a, b = SDis.bounds(secret_distribution)
            if a == -1 and b == 1:
                return True

        return False

    @staticmethod
    def is_binary(secret_distribution):
        """Return true if the secret is binary (sparse or not)

        :param secret_distribution: distribution of secret, see module level documentation for details

        EXAMPLES::

            sage: from estimator import SDis
            sage: SDis.is_binary(False)
            False

            sage: SDis.is_binary(True)
            False

            sage: SDis.is_binary(0.1)
            False

            sage: SDis.is_binary(0.12)
            False

            sage: SDis.is_binary(((-1, 1), 64))
            False

            sage: SDis.is_binary((-1, 1))
            False

            sage: SDis.is_binary(((0, 1), 64))
            True

            sage: SDis.is_binary((0, 1))
            True

        """

        if SDis.is_bounded_uniform(secret_distribution):
            a, b = SDis.bounds(secret_distribution)
            if a == 0 and b == 1:
                return True

        return False

    @staticmethod
    def nonzero(secret_distribution, n):
        """Return number of non-zero elements or ``None``

        :param secret_distribution: distribution of secret, see module level documentation for details
        :param n: LWE dimension `n > 0`

        """
        try:
            (a, b) = secret_distribution
            try:
                (a, b), h = (a, b)  # noqa
                return h
            except (TypeError, ValueError):
                if n is None:
                    raise ValueError("Parameter n is required for sparse secrets.")
                B = ZZ(b - a + 1)
                h = round((B - 1) / B * n)
                return h
        except (TypeError, ValueError):
            raise ValueError("Cannot extract `h`.")

    @staticmethod
    def mean(secret_distribution, q=None, n=None):
        """
        Mean of the secret per component.

        :param secret_distribution: distribution of secret, see module level documentation for details
        :param n: only used for sparse secrets

        EXAMPLE::

            sage: from estimator import SDis
            sage: SDis.mean(True)
            0

            sage: SDis.mean(False, q=10)
            0

            sage: SDis.mean(0.1)
            0

            sage: SDis.mean(0.12)
            0

            sage: SDis.mean(((-3,3)))
            0

            sage: SDis.mean(((-3,3),64), n=256)
            0

            sage: SDis.mean(((-3,2)))
            -1/2

            sage: SDis.mean(((-3,2),64), n=256)
            -3/20

        """
        if not SDis.is_small(secret_distribution):
            # uniform distribution variance
            if q is None:
                raise ValueError("Parameter q is required for uniform secret.")
            a = -floor(q / 2)
            b = floor(q / 2)
            return (a + b) / ZZ(2)
        else:
            try:
                a, b = SDis.bounds(secret_distribution)

                try:
                    (a, b), h = secret_distribution
                    if n is None:
                        raise ValueError("Parameter n is required for sparse secrets.")
                    return h / ZZ(n) * (b * (b + 1) - a * (a - 1)) / (2 * (b - a))
                except (TypeError, ValueError):
                    return (a + b) / ZZ(2)
            except ValueError:
                # small with no bounds, it's normal
                return ZZ(0)

    @staticmethod
    def variance(secret_distribution, alpha=None, q=None, n=None):
        """
        Variance of the secret per component.

        :param secret_distribution: distribution of secret, see module level documentation for details
        :param alpha: only used for normal form LWE
        :param q: only used for normal form LWE
        :param n: only used for sparse secrets

        EXAMPLE::

            sage: from estimator import SDis, alphaf
            sage: SDis.variance(True, 8./2^15, 2^15).sqrt().n()
            3.19...

            sage: from estimator import SDis
            sage: SDis.variance("normal", 8./2^15, 2^15).sqrt().n()
            3.19...

            sage: SDis.variance(alphaf(1.0, 100, True), q=100)
            1.00000000000000

            sage: SDis.variance(alphaf(1.2, 100, True), q=100)
            1.44000000000000

            sage: SDis.variance((-3,3), 8./2^15, 2^15)
            4

            sage: SDis.variance(((-3,3),64), 8./2^15, 2^15, n=256)
            7/6

            sage: SDis.variance((-3,2))
            35/12

            sage: SDis.variance(((-3,2),64), n=256)
            371/400

            sage: SDis.variance((-1,1), 8./2^15, 2^15)
            2/3

            sage: SDis.variance(((-1,1),64), 8./2^15, 2^15, n=256)
            1/4

        ..  note :: This function assumes that the bounds are of opposite sign, and that the
            distribution is centred around zero.
        """
        if not SDis.is_small(secret_distribution):
            # uniform distribution variance
            a = -floor(q / 2)
            b = floor(q / 2)
            n = b - a + 1
            return (n ** 2 - 1) / ZZ(12)
        else:
            try:
                a, b = SDis.bounds(secret_distribution)
            except ValueError:
                # small with no bounds, it's normal
                if isinstance(secret_distribution, bool) or secret_distribution == "normal":
                    return stddevf(alpha * q) ** 2
                else:
                    return RR(stddevf(secret_distribution * q))**2
            try:
                (a, b), h = secret_distribution
            except (TypeError, ValueError):
                return ((b - a + 1) ** 2 - 1) / ZZ(12)
            if n is None:
                raise ValueError("Parameter n is required for sparse secrets.")
            if not (a <= 0 and 0 <= b):
                raise ValueError("a <= 0 and 0 <= b is required for uniform bounded secrets.")
            # E(x^2), using https://en.wikipedia.org/wiki/Square_pyramidal_number
            tt = (h / ZZ(n)) * (2 * b ** 3 + 3 * b ** 2 + b - 2 * a ** 3 + 3 * a ** 2 - a) / (ZZ(6) * (b - a))
            # Var(x) = E(x^2) - E(x)^2
            return tt - SDis.mean(secret_distribution, n=n) ** 2


def switch_modulus(f, n, alpha, q, secret_distribution, *args, **kwds):
    """

    :param f: run f
    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details


    """
    length = SDis.nonzero(secret_distribution, n)
    s_var = SDis.variance(secret_distribution, alpha, q, n=n)

    p = RR(ceil(sqrt(2 * pi * s_var * length / ZZ(12)) / alpha))
    if p < 32:  # some random point
        # we can't pretend everything is uniform any more, p is too small
        p = RR(ceil(sqrt(2 * pi * s_var * length * 2 / ZZ(12)) / alpha))
    beta = RR(sqrt(2) * alpha)
    return f(n, beta, p, secret_distribution, *args, **kwds)


# Repetition


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
    :param sigma: (lists of) Gaussian width parameters
    :param q: modulus `0 < q`

    """
    try:
        sigma = sum(sigma_ ** 2 for sigma_ in sigma).sqrt()
    except TypeError:
        pass
    RR = parent(sigma)
    advantage = RR(exp(-RR(pi) * (RR(sigma / q) ** 2)))
    # return target_advantage/advantage**2  # old
    return amplify(target_advantage, advantage, majority=True)


def rinse_and_repeat(
    f,
    n,
    alpha,
    q,
    success_probability=0.99,
    m=oo,
    optimisation_target=u"red",
    decision=True,
    repeat_select=None,
    *args,
    **kwds
):
    """Find best trade-off between success probability and running time.

    :param f:                    a function returning a cost estimate
    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param success_probability: targeted success probability < 1
    :param optimisation_target:  which value to minimise
    :param decision:             set if `f` solves a decision problem, unset for search problems
    :param repeat_select:        passed through to ``cost_repeat`` as parameter ``select``
    :param samples:              the number of available samples

    """
    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)
    RR = parent(alpha)

    best = None
    step_size = 32
    i = floor(-log(success_probability, 2))
    has_solution = False
    while True:
        prob = RR(min(2 ** -i, success_probability))
        try:
            current = f(n, alpha, q, success_probability=prob, m=m, *args, **kwds)
            repeat = amplify(success_probability, prob, majority=decision)
            # TODO does the following hold for all algorithms?
            current = current.repeat(repeat, select=repeat_select)
            has_solution = True
        except (OutOfBoundsError, InsufficientSamplesError):
            key = list(best)[0] if best is not None else optimisation_target
            current = Cost()
            current[key] = PlusInfinity()
        current["epsilon"] = ZZ(2) ** -i

        logging.getLogger("repeat").debug(current)

        key = list(current)[0]
        if best is None:
            if current[key] is not PlusInfinity():
                best = current
            i += step_size

            if i > 8192:  # somewhat arbitrary constant
                raise RuntimeError("Looks like we are stuck in an infinite loop.")

            continue

        if key not in best or current[key] < best[key]:
            best = current
            i += step_size
        else:
            # we go back
            i = -log(best["epsilon"], 2) - step_size
            i += step_size // 2
            if i <= 0:
                i = step_size // 2
            # and half the step size
            step_size = step_size // 2

        if step_size == 0:
            break

    if not has_solution:
        raise RuntimeError("No solution found for chosen parameters.")

    return best


class BKZ:
    """
    Cost estimates for BKZ.
    """

    @staticmethod
    def _delta_0f(k):
        """
        Compute `δ_0` from block size `k` without enforcing `k` in ZZ.

        δ_0 for k≤40 were computed as follows:

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
                delta_0 = []
                t = []
                i = 0
    !!            while i < trials:
                    threads = int(open("delta_0.nthreads").read()) # make sure this file exists
                    pool = Pool(threads)
                    A = [(IntegerMatrix.random(d, "qary", k=d//2, bits=50), beta) for j in range(threads)]
                    for (t_, delta_0_) in pool.imap_unordered(f, A):
                        t.append(t_)
                        delta_0.append(delta_0_)
                    i += threads
                    print u"β: %2d, δ_0: %.5f, time: %5.1fs, (%2d,%2d)"%(beta, mean(delta_0), mean(t), i, threads)
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

        if k <= 2:
            return RR(1.0219)
        elif k < 40:
            for i in range(1, len(small)):
                if small[i][0] > k:
                    return RR(small[i - 1][1])
        elif k == 40:
            return RR(small[-1][1])
        else:
            return RR(k / (2 * pi * e) * (pi * k) ** (1 / k)) ** (1 / (2 * (k - 1)))

    @staticmethod
    def _beta_secant(delta):
        """
        Estimate required blocksize `k` for a given root-hermite factor δ based on [PhD:Chen13]_

        :param delta: root-hermite factor

        EXAMPLE::

            sage: from estimator import BKZ
            sage: 50 == BKZ._beta_secant(1.0121)
            True
            sage: 100 == BKZ._beta_secant(1.0093)
            True
            sage: BKZ._beta_secant(1.0024) # Chen reports 800
            808

        .. [PhD:Chen13] Yuanmi Chen. Réduction de réseau et sécurité concrète du chiffrement
                        complètement homomorphe. PhD thesis, Paris 7, 2013.
        """
        # newton() will produce a "warning", if two subsequent function values are
        # indistinguishable (i.e. equal in terms of machine precision). In this case
        # newton() will return the value k in the middle between the two values
        # k1,k2 for which the function values were indistinguishable.
        # Since f approaches zero for k->+Infinity, this may be the case for very
        # large inputs, like k=1e16.
        # For now, these warnings just get printed and the value k is used anyways.
        # This seems reasonable, since for such large inputs the exact value of k
        # doesn't make such a big difference.
        try:
            k = newton(lambda k: RR(BKZ._delta_0f(k) - delta), 100, fprime=None, args=(), tol=1.48e-08, maxiter=500)
            k = ceil(k)
            if k < 40:
                # newton may output k < 40. The old beta method wouldn't do this. For
                # consistency, call the old beta method, i.e. consider this try as "failed".
                raise RuntimeError("k < 40")
            return k
        except (RuntimeError, TypeError):
            # if something fails, use old beta method
            k = BKZ._beta_simple(delta)
            return k

    @staticmethod
    def _beta_find_root(delta):
        """

        TESTS::

            sage: from estimator import betaf, delta_0f
            sage: betaf(delta_0f(500))
            500

        """
        # handle k < 40 separately
        k = ZZ(40)
        if delta_0f(k) < delta:
            return k

        try:
            k = find_root(lambda k: RR(BKZ._delta_0f(k) - delta), 40, 2 ** 16, maxiter=500)
            k = ceil(k - 1e-8)
        except RuntimeError:
            # finding root failed; reasons:
            # 1. maxiter not sufficient
            # 2. no root in given interval
            k = BKZ._beta_simple(delta)
        return k

    @staticmethod
    def _beta_simple(delta):
        """
        Estimate required blocksize `k` for a given root-hermite factor δ based on [PhD:Chen13]_

        :param delta: root-hermite factor

        EXAMPLE::

            sage: from estimator import betaf
            sage: 50 == betaf(1.0121)
            True
            sage: 100 == betaf(1.0093)
            True
            sage: betaf(1.0024) # Chen reports 800
            808

        .. [PhD:Chen13] Yuanmi Chen. Réduction de réseau et sécurité concrète du chiffrement
                        complètement homomorphe. PhD thesis, Paris 7, 2013.
        """
        k = ZZ(40)

        while delta_0f(2 * k) > delta:
            k *= 2
        while delta_0f(k + 10) > delta:
            k += 10
        while True:
            if delta_0f(k) < delta:
                break
            k += 1

        return k

    @staticmethod
    def svp_repeat(beta, d):
        """Return number of SVP calls in BKZ-β

        :param beta: block size
        :param d: dimension

        .. note :: loosely based on experiments in [PhD:Chen13]

        """
        return 8 * d

    @staticmethod
    @cached_function
    def GSA(n, q, delta, m):
        """
        Compute the Gram-Schmidt lengths based on the GSA.

        :param n: LWE dimension `n > 0`
        :param q: modulus `0 < q`
        :param delta: root-Hermite factor
        :param m: lattice dimension

        .. [RSA:LinPei11] Richard Lindner and Chris Peikert. Better key sizes (and attacks) for LWE-based encryption.
                          In Aggelos Kiayias, editor, CT-RSA 2011, volume 6558 of LNCS, pages 319–339. Springer,
                          February 2011.
        """
        log_delta = RDF(log(delta))
        log_q = RDF(log(q))
        qnm = log_q * (n / m)
        qnm_p_log_delta_m = qnm + log_delta * m
        tmm1 = RDF(2 * m / (m - 1))
        b = [(qnm_p_log_delta_m - log_delta * (tmm1 * i)) for i in range(m)]
        b = [log_q - b[-1 - i] for i in range(m)]
        b = list(map(lambda x: x.exp(), b))
        return b

    # BKZ Estimates

    @staticmethod
    def LLL(d, B=None):
        """
        Runtime estimation for LLL algorithm

        :param d: lattice dimension
        :param B: bit-size of entries

        ..  [CheNgu11] Chen, Y., & Nguyen, P.  Q.  (2011).  BKZ 2.0: better lattice security
            estimates.  In D.  H.  Lee, & X.  Wang, ASIACRYPT~2011 (pp.  1–20).  : Springer,
            Heidelberg.
        """
        if B:
            return d ** 3 * B ** 2
        else:
            return d ** 3  # ignoring B for backward compatibility

    @staticmethod
    def LinPei11(beta, d, B=None):
        """
        Runtime estimation assuming the Lindner-Peikert model in elementary operations.

        ..  [LinPei11] Lindner, R., & Peikert, C.  (2011).  Better key sizes (and attacks) for LWE-based
            encryption.  In A.  Kiayias, CT-RSA~2011 (pp.  319–339).  : Springer, Heidelberg.

        :param beta: block size
        :param d: lattice dimension
        :param B: bit-size of entries

        """
        delta_0 = delta_0f(beta)
        return BKZ.LLL(d, B) + ZZ(2) ** RR(1.8 / log(delta_0, 2) - 110 + log(2.3 * 10 ** 9, 2))

    @staticmethod
    def _BDGL16_small(beta, d, B=None):
        u"""
         Runtime estimation given `β` and assuming sieving is used to realise the SVP oracle for small dimensions.

         :param beta: block size
         :param d: lattice dimension
         :param B: bit-size of entries

        ..  [BDGL16] Becker, A., Ducas, L., Gama, N., & Laarhoven, T.  (2016).  New directions in
        nearest neighbor searching with applications to lattice sieving.  In SODA 2016, (pp.
        10–24).

         """
        return BKZ.LLL(d, B) + ZZ(2) ** RR(0.292 * beta + 16.4 + log(BKZ.svp_repeat(beta, d), 2))

    @staticmethod
    def _BDGL16_asymptotic(beta, d, B=None):
        u"""
         Runtime estimation given `β` and assuming sieving is used to realise the SVP oracle.

         :param beta: block size
         :param d: lattice dimension
         :param B: bit-size of entries

        ..  [BDGL16] Becker, A., Ducas, L., Gama, N., & Laarhoven, T.  (2016).  New directions in
        nearest neighbor searching with applications to lattice sieving.  In SODA 2016, (pp.
        10–24).

        """
        # TODO we simply pick the same additive constant 16.4 as for the experimental result in [BDGL16]
        return BKZ.LLL(d, B) + ZZ(2) ** RR(0.292 * beta + 16.4 + log(BKZ.svp_repeat(beta, d), 2))

    @staticmethod
    def BDGL16(beta, d, B=None):
        u"""
         Runtime estimation given `β` and assuming sieving is used to realise the SVP oracle.

         :param beta: block size
         :param d: lattice dimension
         :param B: bit-size of entries

        ..  [BDGL16] Becker, A., Ducas, L., Gama, N., & Laarhoven, T.  (2016).  New directions in
            nearest neighbor searching with applications to lattice sieving.  In SODA 2016, (pp.
            10–24).

        """
        
            return BKZ._BDGL16_asymptotic(beta, d, B)

    @staticmethod
    def LaaMosPol14(beta, d, B=None):
        u"""
         Runtime estimation for quantum sieving.

         :param beta: block size
         :param d: lattice dimension
         :param B: bit-size of entries

         ..  [LaaMosPol14] Thijs Laarhoven, Michele Mosca, & Joop van de Pol.  Finding shortest
             lattice vectors faster using quantum search.  Cryptology ePrint Archive, Report
             2014/907, 2014.  https://eprint.iacr.org/2014/907.

         ..  [Laarhoven15] Laarhoven, T.  (2015).  Search problems in cryptography: from
             fingerprinting to lattice sieving (Doctoral dissertation).  Eindhoven University of
             Technology. http://repository.tue.nl/837539
         """
        return BKZ.LLL(d, B) + ZZ(2) ** RR((0.265 * beta + 16.4 + log(BKZ.svp_repeat(beta, d), 2)))

    @staticmethod
    def CheNgu12(beta, d, B=None):
        u"""
         Runtime estimation given `β` and assuming [CheNgu12]_ estimates are correct.

         :param beta: block size
         :param d: lattice dimension
         :param B: bit-size of entries

         The constants in this function were derived as follows based on Table 4 in
         [CheNgu12]_::

             sage: dim = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
             sage: nodes = [39.0, 44.0, 49.0, 54.0, 60.0, 66.0, 72.0, 78.0, 84.0, 96.0, 99.0, 105.0, 111.0, 120.0, 127.0, 134.0]  # noqa
             sage: times = [c + log(200,2).n() for c in nodes]
             sage: T = list(zip(dim, nodes))
             sage: var("a,b,c,k")
             (a, b, c, k)
             sage: f = a*k*log(k, 2.0) + b*k + c
             sage: f = f.function(k)
             sage: f.subs(find_fit(T, f, solution_dict=True))
             k |--> 0.270188776350190*k*log(k) - 1.0192050451318417*k + 16.10253135200765

         The estimation

         2^(0.270188776350190*beta*log(beta) - 1.0192050451318417*beta + 16.10253135200765)

         is of the number of enumeration nodes, hence we need to multiply by the number of
         cycles to process one node.  This cost per node is typically estimated as 100 [FPLLL].

         ..  [CheNgu12] Yuanmi Chen and Phong Q.  Nguyen.  BKZ 2.0: Better lattice security
         estimates (Full Version). 2012. http://www.di.ens.fr/~ychen/research/Full_BKZ.pdf

         ..  [FPLLL] The FPLLL development team.  fplll, a lattice reduction library.  2016.
         Available at https://github.com/fplll/fplll
         """
        # TODO replace these by fplll timings
        repeat = BKZ.svp_repeat(beta, d)
        cost = RR(0.270188776350190 * beta * log(beta) - 1.0192050451318417 * beta + 16.10253135200765 + log(100, 2))
        return BKZ.LLL(d, B) + repeat * ZZ(2) ** cost

    @staticmethod
    def ABFKSW20(beta, d, B=None):
        """
        Enumeration cost according to [ABFKSW20]_.

        :param beta: block size
        :param d: lattice dimension
        :param B: bit-size of entries

        ..  [ABFKSW20] Martin R. Albrecht, Shi Bai, Pierre-Alain Fouque, Paul Kirchner, Damien
        Stehlé and Weiqiang Wen.  Faster Enumeration-based Lattice Reduction: Root Hermite
        Factor $k^{1/(2k)}$ in Time $k^{k/8 + o(k)}$.  CRYPTO 2020.
        """
        if 1.5 * beta >= d or beta <= 92:  # 1.5β is a bit arbitrary, β≤92 is the crossover point
            cost = RR(0.1839 * beta * log(beta, 2) - 0.995 * beta + 16.25 + log(100, 2))
        else:
            cost = RR(0.125 * beta * log(beta, 2) - 0.547 * beta + 10.4 + log(100, 2))
        repeat = BKZ.svp_repeat(beta, d)

        return BKZ.LLL(d, B) + repeat * ZZ(2) ** cost

    @staticmethod
    def ADPS16(beta, d, B=None, mode="classical"):
        u"""
         Runtime estimation from [ADPS16]_.

         :param beta: block size
         :param n: LWE dimension `n > 0`
         :param B: bit-size of entries

         EXAMPLE::

             sage: from estimator import BKZ, Param, dual, partial
             sage: cost_model = partial(BKZ.ADPS16, mode="paranoid")
             sage: dual(*Param.LindnerPeikert(128), reduction_cost_model=cost_model)
                 rop:   2^37.3
                   m:      346
                 red:   2^37.3
             delta_0: 1.008209
                beta:      127
                   d:      346
                 |v|:  284.363
              repeat:     2000
             epsilon: 0.062500

         ..  [ADPS16] Edem Alkim, Léo Ducas, Thomas Pöppelmann, & Peter Schwabe (2016).
             Post-quantum key exchange - A New Hope.  In T. Holz, & S. Savage, 25th USENIX
             Security Symposium, USENIX Security 16 (pp. 327–343). USENIX Association.
         """

        if mode not in ("classical", "quantum", "paranoid"):
            raise ValueError("Mode '%s' not understood" % mode)

        c = {
            "classical": 0.2920,
            "quantum": 0.2650,  # paper writes 0.262 but this isn't right, see above
            "paranoid": 0.2075,
        }

        c = c[mode]

        return ZZ(2) ** RR(c * beta)

    sieve = BDGL16
    qsieve = LaaMosPol14
    lp = LinPei11
    enum = ABFKSW20


def delta_0f(beta):
    """
    Compute root-Hermite factor `δ_0` from block size `β`.
    """
    beta = ZZ(round(beta))
    return BKZ._delta_0f(beta)


def betaf(delta):
    """
    Compute block size `β` from root-Hermite factor `δ_0`.
    """
    # TODO: decide for one strategy (secant, find_root, old) and its error handling
    k = BKZ._beta_find_root(delta)
    return k


reduction_default_cost = BKZ.enum


def lattice_reduction_cost(cost_model, delta_0, d, B=None):
    """
    Return cost dictionary for returning vector of norm` δ_0^d Vol(Λ)^{1/d}` using provided lattice
    reduction algorithm.

    :param lattice_reduction_estimate:
    :param delta_0: root-Hermite factor `δ_0 > 1`
    :param d: lattice dimension
    :param B: bit-size of entries

    """
    beta = betaf(delta_0)
    cost = cost_model(beta, d, B)
    return Cost([("rop", cost), ("red", cost), ("delta_0", delta_0), ("beta", beta)])


def lattice_reduction_opt_m(n, q, delta):
    """
    Return the (heuristically) optimal lattice dimension `m`

    :param n: LWE dimension `n > 0`
    :param q: modulus `0 < q`
    :param delta: root Hermite factor `δ_0`

    """
    # round can go wrong if the input is not a floating point number
    return ZZ(round(sqrt(n * log(q, 2) / log(delta, 2)).n()))


def sieve_or_enum(func):
    """
    Take minimum of sieving or enumeration for lattice-based attacks.

    :param func: a lattice-reduction based estimator
    """

    def wrapper(*args, **kwds):
        from copy import copy

        kwds = copy(kwds)

        a = func(*args, reduction_cost_model=BKZ.sieve, **kwds)
        b = func(*args, reduction_cost_model=BKZ.enum, **kwds)
        if a["red"] <= b["red"]:
            return a
        else:
            return b

    return wrapper


# Combinatorial Algorithms for Sparse/Sparse Secrets


def guess_and_solve(f, n, alpha, q, secret_distribution, success_probability=0.99, **kwds):
    """Guess components of the secret.

    :param f:
    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param success_probability: targeted success probability < 1

    EXAMPLE::

        sage: from estimator import guess_and_solve, dual_scale, partial
        sage: q = next_prime(2^30)
        sage: n, alpha = 512, 8/q
        sage: dualg = partial(guess_and_solve, dual_scale)
        sage: dualg(n, alpha, q, secret_distribution=((-1,1), 64))
            rop:   2^63.9
              m:      530
            red:   2^63.9
        delta_0: 1.008803
           beta:      111
         repeat:   2^21.6
              d:     1042
              c:    9.027
              k:        0

    """

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)
    RR = parent(alpha)

    a, b = SDis.bounds(secret_distribution)
    h = SDis.nonzero(secret_distribution, n)

    size = b - a + 1
    best = None
    step_size = 16
    fail_attempts, max_fail_attempts = 0, 5
    while step_size >= n:
        step_size //= 2
    i = 0
    while True:
        if i < 0:
            break
        try:
            current = f(
                n - i,
                alpha,
                q,
                secret_distribution=secret_distribution,
                success_probability=max(1 - 1 / RR(2) ** 80, success_probability),
                **kwds
            )
        except (OutOfBoundsError, RuntimeError, InsufficientSamplesError):
            i += abs(step_size)
            fail_attempts += 1
            if fail_attempts > max_fail_attempts:
                break
            continue
        if h is None or i < h:
            repeat = size ** i
        else:
            # TODO: this is too pessimistic
            repeat = (size) ** h * binomial(i, h)
        current["k"] = i
        current = current.repeat(repeat, select={"m": False})

        logging.getLogger("binsearch").debug(current)

        key = list(current)[0]
        if best is None:
            best = current
            i += step_size
        else:
            if best[key] > current[key]:
                best = current
                i += step_size
            else:
                step_size = -1 * step_size // 2
                i += step_size

        if step_size == 0:
            break
    if best is None:
        raise RuntimeError("No solution could be found.")
    return best


def success_probability_drop(n, h, k, fail=0, rotations=False):
    """
    Probability that `k` randomly sampled components have ``fail`` non-zero components amongst
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


def drop_and_solve(
    f,
    n,
    alpha,
    q,
    secret_distribution=True,
    success_probability=0.99,
    postprocess=True,
    decision=True,
    rotations=False,
    **kwds
):
    """
    Solve instances of dimension ``n-k`` with increasing ``k`` using ``f`` and pick parameters such
    that cost is minimised.

    :param f: attack estimate function
    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param success_probability: targeted success probability < 1
    :param postprocess: check against shifted distributions
    :param decision: the underlying algorithm solves the decision version or not

    EXAMPLE::

        sage: from estimator import drop_and_solve, dual_scale, primal_usvp, partial
        sage: q = next_prime(2^30)
        sage: n, alpha = 512, 8/q
        sage: primald = partial(drop_and_solve, primal_usvp)
        sage: duald = partial(drop_and_solve, dual_scale)

        sage: duald(n, alpha, q, secret_distribution=((-1,1), 64))
                rop:   2^55.9
                  m:      478
                red:   2^55.3
            delta_0: 1.009807
               beta:       89
             repeat:   2^14.2
                  d:      920
                  c:    8.387
                  k:       70
        postprocess:        8

        sage: duald(n, alpha, q, secret_distribution=((-3,3), 64))
                rop:   2^59.8
                  m:      517
                red:   2^59.7
            delta_0: 1.009403
               beta:       97
             repeat:   2^18.2
                  d:      969
                  c:    3.926
                  k:       60
        postprocess:        6

        sage: kwds = {"use_lll":False, "postprocess":False}
        sage: duald(n, alpha, q, secret_distribution=((-1,1), 64), **kwds)
                rop:   2^69.7
                  m:      521
                red:   2^69.7
            delta_0: 1.008953
               beta:      107
             repeat:      257
                  d:     1033
                  c:    9.027
                  k:        0
        postprocess:        0

        sage: duald(n, alpha, q, secret_distribution=((-3,3), 64), **kwds)
                rop:   2^74.2
                  m:      560
                red:   2^74.2
            delta_0: 1.008668
               beta:      114
             repeat:  524.610
                  d:     1068
                  c:    4.162
                  k:        4
        postprocess:        0

        sage: duald(n, alpha, q, secret_distribution=((-3,3), 64), rotations=True, **kwds)
        Traceback (most recent call last):
          ...
        ValueError: Rotations are only support as part of the primal-usvp attack on NTRU.

        sage: primald(n, alpha, q, secret_distribution=((-3,3), 64), rotations=True, **kwds)
                rop:   2^51.5
                red:   2^51.5
            delta_0: 1.010046
               beta:       84
                  d:      913
                  m:      444
             repeat: 1.509286
                  k:       44
        postprocess:        0

        sage: primald(n, alpha, q, secret_distribution=((-3,3), 64), rotations=False, **kwds)
                rop:   2^58.2
                red:   2^58.2
            delta_0: 1.009350
               beta:       98
                  d:     1001
                  m:      492
             repeat: 1.708828
                  k:        4
        postprocess:        0

    This function is based on:

    ..  [Albrecht17] Albrecht, M.  R.  (2017).  On dual lattice attacks against small-secret LWE and
        parameter choices in helib and SEAL.  In J.  Coron, & J.  B.  Nielsen, EUROCRYPT 2017, Part II
        (pp.  103–129).


    """

    if rotations and f.__name__ != "primal_usvp":
        raise ValueError("Rotations are only support as part of the primal-usvp attack on NTRU.")

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)

    best = None

    if not decision and postprocess:
        raise ValueError("Postprocessing is only defined for the dual attack which solves the decision version.")

    # too small a step size leads to an early abort, too large a step
    # size means stepping over target
    step_size = int(n // 32)

    if not SDis.is_bounded_uniform(secret_distribution):
        raise NotImplementedError("Only bounded uniform secrets are currently supported.")

    a, b = SDis.bounds(secret_distribution)
    assert SDis.is_bounded_uniform(secret_distribution)
    h = SDis.nonzero(secret_distribution, n)

    k = ZZ(0)

    while (n - h) > k:
        probability = success_probability_drop(n, h, k, rotations=rotations)

        # increase precision until the probability is meaningful
        while success_probability ** probability == 1:
            success_probability = RealField(64 + success_probability.prec())(success_probability)

        current = f(
            n - k,
            alpha,
            q,
            success_probability=success_probability ** probability if decision else success_probability,
            secret_distribution=secret_distribution,
            **kwds
        )

        cost_lat = list(current.values())[0]
        cost_post = 0
        if postprocess:
            repeat = current["repeat"]
            dim = current["d"]
            for i in range(1, k):
                # compute inner products with rest of A
                cost_post_i = 2 * repeat * dim * k
                # there are (k)(i) positions and max(s_i)-min(s_i) options per position
                # for each position we need to add/subtract the right elements
                cost_post_i += repeat * binomial(k, i) * (b - a) ** i * i
                if cost_post + cost_post_i >= cost_lat:
                    postprocess = i
                    break
                cost_post += cost_post_i
                probability += success_probability_drop(n, h, k, i, rotations=rotations)

        current["rop"] = cost_lat + cost_post
        current = current.repeat(1 / probability, select={"m": False})
        current["k"] = k
        current["postprocess"] = postprocess
        current = current.reorder(["rop"])

        logging.getLogger("guess").debug(current)

        key = list(current)[0]
        if best is None:
            best = current
            k += step_size
            continue

        if current[key] < best[key]:
            best = current
            k += step_size
        else:
            # we go back
            k = best["k"] - step_size
            k += step_size // 2
            if k <= 0:
                k = step_size // 2
            # and half the step size
            step_size = step_size // 2

        if step_size == 0:
            break

    return best


# Primal Attack (uSVP)


def _primal_scale_factor(secret_distribution, alpha=None, q=None, n=None):
    """
    Scale factor for primal attack, in the style of [BaiGal14]. In the case of non-centered secret
    distributions, it first appropriately rebalances the lattice bases to maximise the scaling.

    :param secret_distribution: distribution of secret, see module level documentation for details
    :param alpha: noise rate `0 ≤ α < 1`, noise has standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param n: only used for sparse secrets

    EXAMPLE::

        sage: from estimator import _primal_scale_factor, alphaf
        sage: _primal_scale_factor(True, 8./2^15, 2^15)
        1.000000000...

        sage: _primal_scale_factor(alphaf(sqrt(2), 2^13, True), \
                                   alpha=alphaf(sqrt(16/3), 2^13, True), q=2^13)
        1.63299316185545

        sage: _primal_scale_factor((-1,1), alpha=8./2^15, q=2^15)
        3.908820095...

        sage: _primal_scale_factor(((-1,1), 64), alpha=8./2^15, q=2^15, n=256)
        6.383076486...

        sage: _primal_scale_factor((-3,3), alpha=8./2^15, q=2^15)
        1.595769121...

        sage: _primal_scale_factor(((-3,3), 64), alpha=8./2^15, q=2^15, n=256)
        2.954790254...

        sage: _primal_scale_factor((-3,2), alpha=8./2^15, q=2^15)
        1.868773442...

        sage: _primal_scale_factor(((-3,2), 64), alpha=8./2^15, q=2^15, n=256)
        3.313928192...

        sage: _primal_scale_factor((0, 1), alpha=sqrt(2*pi)/2^15, q=2^15)
        2.000000000...

        sage: _primal_scale_factor((0, 1), alpha=sqrt(2*pi)/2^15/2, q=2^15)
        1.000000000...

        sage: _primal_scale_factor((0, 1), alpha=sqrt(2*pi)/2^15/1.99, q=2^15) > 1
        True

    ..  [BaiGal14] Bai, S., & Galbraith, S.  D.  (2014).  Lattice decoding attacks on binary
        LWE.  In W.  Susilo, & Y.  Mu, ACISP 14 (pp.  322–337).  : Springer, Heidelberg.
    """

    # For small/sparse secret use Bai and Galbraith's scaled embedding
    # NOTE: We assume a <= 0 <= b

    scale = RR(1)
    if SDis.is_small(secret_distribution, alpha=alpha):
        # target same same shortest vector length as in Kannan's embedding of same dimension
        stddev = stddevf(alpha * q)
        var_s = SDis.variance(secret_distribution, alpha, q, n=n)
        if stddev ** 2 > var_s:
            # only scale if the error is sampled wider than the secret
            scale = stddev / RR(sqrt(var_s))

    return scale


def _primal_usvp(
    block_size,
    n,
    alpha,
    q,
    scale=1,
    m=oo,
    success_probability=0.99,
    kannan_coeff=None,
    d=None,
    reduction_cost_model=reduction_default_cost,
):
    """
    Estimate cost of solving LWE using primal attack (uSVP version)

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param scale: The identity part of the lattice basis is scaled by this constant.
    :param m: number of available LWE samples `m > 0`
    :param d: dimension for the attack d <= m + 1  (`None' for optimized choice)
    :parap kannan_coeff: Coeff for Kannan's embedding (`None' to set it kannan_coeff=stddev,
        which is optimal at least when Distrib(secret) = Distrib(Error).)
    :param success_probability: targeted success probability < 1
    :param reduction_cost_model: cost model for lattice reduction

    .. note:: This is the low-level function, in most cases you will want to call ``primal_usvp``

    """

    n, alpha, q = Param.preprocess(n, alpha, q)
    RR = alpha.parent()
    q = RR(q)
    delta_0 = delta_0f(block_size)
    stddev = stddevf(alpha * q)
    block_size = RR(block_size)

    scale = RR(scale)

    m = min(2 * ceil(sqrt(n * log(q) / log(delta_0))), m)

    if kannan_coeff is None:
        kannan_coeff = stddev

    def log_b_star(d):
        return delta_0.log() * (2 * block_size - d) + (kannan_coeff.log() + n * scale.log() + (d - n - 1) * q.log()) / d

    C = (stddev ** 2 * (block_size - 1) + kannan_coeff ** 2).log() / 2

    # find the smallest d \in [n,m] s.t. a*d^2 + b*d + c >= 0
    # if no such d exists, return the upper bound m
    def solve_for_d(n, m, a, b, c):
        if a*n*n + b*n + c >= 0:  # trivial case
            return n

        # solve for ad^2 + bd + c == 0

        disc = b*b - 4*a*c  # the discriminant
        if disc < 0:  # no solution, return m
            return m

        # compute the two solutions
        d1 = (-b + sqrt(disc))/(2*a)
        d2 = (-b - sqrt(disc))/(2*a)
        if a > 0:  # the only possible solution is ceiling(d2)
            return min(m, ceil(d2))

        # the case a<=0:

        # if n is to the left of d1 then the first solution is ceil(d1)
        if n <= d1:
            return min(m, ceil(d1))

        # otherwise, n must be larger than d2 (since an^2+bn+c<0) so no solution
        return m

    # we have m samples → the largest permissible dimension d is m+1
    if d is None:
        #   for d in range(n, m + 2):
        #       if log_b_star(d) - C >= 0:
        #           break
        aa = -delta_0.log()
        bb = delta_0.log()*2*block_size + q.log() - C
        cc = kannan_coeff.log() + n * scale.log() - (n + 1) * q.log()
        #    assert d == solve_for_d(n, m+1, aa, bb, cc)
        d = solve_for_d(n, m+1, aa, bb, cc)

    assert d <= m + 1

    def ineq(d):
        lhs = sqrt(stddev ** 2 * (block_size - 1) + kannan_coeff ** 2)
        rhs = delta_0 ** (2 * block_size - d) * (kannan_coeff * scale ** n * q ** (d - n - 1)) ** (ZZ(1) / d)
        return lhs <= rhs

    ret = lattice_reduction_cost(reduction_cost_model, delta_0, d)
    if not ineq(d):
        ret["rop"] = oo
        ret["red"] = oo

    ret["d"] = d
    ret["delta_0"] = delta_0

    return ret


def primal_usvp(
    n,
    alpha,
    q,
    secret_distribution=True,
    m=oo,
    success_probability=0.99,
    kannan_coeff=None,
    d=None,
    reduction_cost_model=reduction_default_cost,
    **kwds
):
    u"""
    Estimate cost of solving LWE using primal attack (uSVP version)

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param m: number of LWE samples `m > 0`
    :param success_probability: targeted success probability < 1
    :param reduction_cost_model: cost model for lattice reduction

    EXAMPLES::

        sage: from estimator import primal_usvp, Param, BKZ
        sage: n, alpha, q = Param.Regev(256)

        sage: primal_usvp(n, alpha, q)
            rop:  2^145.5
            red:  2^145.5
        delta_0: 1.005374
           beta:      256
              d:      694
              m:      437

        sage: primal_usvp(n, alpha, q, secret_distribution=True, m=n)
            rop:  2^186.9
            red:  2^186.9
        delta_0: 1.004638
           beta:      320
              d:      513
              m:      256

        sage: primal_usvp(n, alpha, q, secret_distribution=False, m=2*n)
            rop:  2^186.9
            red:  2^186.9
        delta_0: 1.004638
           beta:      320
              d:      513
              m:      512

        sage: primal_usvp(n, alpha, q, reduction_cost_model=BKZ.sieve)
            rop:  2^103.6
            red:  2^103.6
        delta_0: 1.005374
           beta:      256
              d:      694
              m:      437

        sage: primal_usvp(n, alpha, q)
            rop:  2^145.5
            red:  2^145.5
        delta_0: 1.005374
           beta:      256
              d:      694
              m:      437

        sage: primal_usvp(n, alpha, q, secret_distribution=(-1,1), m=n)
            rop:   2^84.7
            red:   2^84.7
        delta_0: 1.007345
           beta:      154
              d:      513
              m:      256

        sage: primal_usvp(n, alpha, q, secret_distribution=((-1,1), 64))
            rop:   2^77.1
            red:   2^77.1
        delta_0: 1.007754
           beta:      140
              d:      477
              m:      220

    ..  [USENIX:ADPS16] Alkim, E., Léo Ducas, Thomas Pöppelmann, & Schwabe, P.  (2015).
        Post-quantum key exchange - a new hope.

    ..  [BaiGal14] Bai, S., & Galbraith, S.  D.  (2014).  Lattice decoding attacks on binary
        LWE.  In W.  Susilo, & Y.  Mu, ACISP 14 (pp.  322–337).  : Springer, Heidelberg.

    """

    if m < 1:
        raise InsufficientSamplesError("m=%d < 1" % m)

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)

    # allow for a larger embedding lattice dimension, using Bai and Galbraith's
    if SDis.is_small(secret_distribution, alpha=alpha):
        m += n

    scale = _primal_scale_factor(secret_distribution, alpha, q, n)

    kwds = {
        "n": n,
        "alpha": alpha,
        "q": q,
        "kannan_coeff": kannan_coeff,
        "d": d,
        "reduction_cost_model": reduction_cost_model,
        "m": m,
        "scale": scale,
    }

    cost = binary_search(
        _primal_usvp,
        start=40,
        stop=2 * n,
        param="block_size",
        predicate=lambda x, best: x["red"] <= best["red"],
        **kwds
    )

    for block_size in range(32, cost["beta"] + 1)[::-1]:
        t = _primal_usvp(block_size=block_size, **kwds)
        if t["red"] == oo:
            break
        cost = t

    if SDis.is_small(secret_distribution, alpha=alpha):
        cost["m"] = cost["d"] - n - 1
    else:
        cost["m"] = cost["d"] - 1

    return cost


# Primal Attack (Enumeration)


def enumeration_cost(n, alpha, q, success_probability, delta_0, m, clocks_per_enum=2 ** 15.1):
    """
    Estimates the cost of performing enumeration.

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param success_probability: target success probability
    :param delta_0: root-Hermite factor `δ_0 > 1`
    :param m: number of LWE samples `m > 0`
    :param clocks_per_enum:      the log of the number of clock cycles needed per enumeration
    """
    target_success_probability = success_probability

    try:
        alpha.is_NaN()
        RR = alpha.parent()

    except AttributeError:
        RR = RealField(128)

    step = RDF(1)

    B = BKZ.GSA(n, q, delta_0, m)

    d = [RDF(1)] * m
    bd = [d[i] * B[i] for i in range(m)]
    scaling_factor = RDF(sqrt(pi) / (2 * alpha * q))
    probs_bd = [RDF((bd[i] * scaling_factor)).erf() for i in range(m)]
    success_probability = prod(probs_bd)

    if RR(success_probability).is_NaN() or success_probability == 0.0:
        # try in higher precision
        step = RR(step)
        d = [RR(1)] * m
        bd = [d[i] * B[i] for i in range(m)]
        scaling_factor = RR(sqrt(pi) / (2 * alpha * q))
        probs_bd = [RR((bd[i] * scaling_factor)).erf() for i in range(m)]
        success_probability = prod(probs_bd)

        if success_probability.is_NaN():
            return OrderedDict([("babai", oo), ("babai_op", oo)])

    if success_probability <= 0:
        # TODO this seems like the wrong except to raise
        raise InsufficientSamplesError("success probability <= 0.")

    bd = map(list, zip(bd, range(len(bd))))
    bd = sorted(bd)

    import bisect

    last_success_probability = success_probability

    while success_probability < RDF(target_success_probability):
        v, i = bd.pop(0)
        d[i] += step
        v += B[i] * step
        last_success_probability = success_probability
        success_probability /= probs_bd[i]
        probs_bd[i] = (v * scaling_factor).erf()
        success_probability *= probs_bd[i]
        bisect.insort_left(bd, [v, i])

        if success_probability == 0 or last_success_probability >= success_probability:
            return Cost([("babai", oo), ("babai_op", oo)])

    r = Cost([("babai", RR(prod(d))), ("babai_op", RR(prod(d)) * clocks_per_enum)])

    return r


def _primal_decode(
    n,
    alpha,
    q,
    secret_distribution=True,
    m=oo,
    success_probability=0.99,
    reduction_cost_model=reduction_default_cost,
    clocks_per_enum=2 ** 15.1,
):
    """
    Decoding attack as described in [LinPei11]_.

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param m: number of LWE samples `m > 0`
    :param success_probability: targeted success probability < 1
    :param clocks_per_enum: the number of enumerations computed per clock cycle

    EXAMPLE:

        sage: from estimator import Param, primal_decode
        sage: n, alpha, q = Param.Regev(256)

        sage: primal_decode(n, alpha, q)
             rop:  2^156.5
               m:      464
             red:  2^156.5
         delta_0: 1.005446
            beta:      251
               d:      720
           babai:  2^141.8
        babai_op:  2^156.9
          repeat:   2^14.2
         epsilon:  2^-12.0

        sage: primal_decode(n, alpha, q, secret_distribution=(-1,1), m=n)
             rop:  2^194.6
               m:      256
             red:  2^194.6
         delta_0: 1.005069
            beta:      280
               d:      512
           babai:  2^179.6
        babai_op:  2^194.7
          repeat:   2^34.2
         epsilon:  2^-32.0

        sage: primal_decode(n, alpha, q, secret_distribution=((-1,1), 64))
             rop:  2^156.5
               m:      464
             red:  2^156.5
         delta_0: 1.005446
            beta:      251
               d:      720
           babai:  2^141.8
        babai_op:  2^156.9
          repeat:   2^14.2
         epsilon:  2^-12.0

    ..  [LinPei11] Lindner, R., & Peikert, C.  (2011).  Better key sizes (and attacks) for
        LWE-based encryption.  In A.  Kiayias, CT-RSA~2011 (pp.  319–339).  : Springer, Heidelberg.
    """

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)

    if m < 1:
        raise InsufficientSamplesError("m=%d < 1" % m)

    if SDis.is_small(secret_distribution, alpha=alpha):
        m_ = m + n
    else:
        m_ = m

    RR = alpha.parent()

    delta_0m1 = (
        _dual(n, alpha, q, success_probability=success_probability, secret_distribution=secret_distribution, m=m)[
            "delta_0"
        ]
        - 1
    )

    step = RR(1.05)
    direction = -1

    def combine(enum, bkz):
        current = Cost()
        current["rop"] = enum["babai_op"] + bkz["red"]

        for key in bkz:
            current[key] = bkz[key]
        for key in enum:
            current[key] = enum[key]
        current[u"m"] = m_ - n if SDis.is_small(secret_distribution, alpha=alpha) else m_
        return current

    depth = 6
    current = None
    while True:
        delta_0 = 1 + delta_0m1

        if delta_0 >= 1.0219 and current is not None:  # LLL is enough
            break

        m_optimal = lattice_reduction_opt_m(n, q, delta_0)
        m_ = min(m_optimal, m_)
        bkz = lattice_reduction_cost(reduction_cost_model, delta_0, m_, B=log(q, 2))
        bkz["d"] = m_

        enum = enumeration_cost(n, alpha, q, success_probability, delta_0, m_, clocks_per_enum=clocks_per_enum)
        current = combine(enum, bkz).reorder(["rop", "m"])

        # if lattice reduction is cheaper than enumration, make it more expensive
        if current["red"] < current["babai_op"]:
            prev_direction = direction
            direction = -1
            if direction != prev_direction:
                step = 1 + RR(step - 1) / 2
            delta_0m1 /= step
        elif current["red"] > current["babai_op"]:
            prev_direction = direction
            direction = 1
            delta_0m1 *= step
        else:
            break
        if direction != prev_direction:
            step = 1 + RR(step - 1) / 2
            depth -= 1
        if depth == 0:
            break

    return current


primal_decode = partial(rinse_and_repeat, _primal_decode, decision=False, repeat_select={"m": False})


# Dual Attack


def _dual_scale_factor(secret_distribution, alpha=None, q=None, n=None, c=None):
    """
    Scale factor for dual attack. Assumes the secret vector has been "re-balanced", as to
    be distributed around 0.

    :param secret_distribution: distribution of secret, see module level documentation for details
    :param alpha: noise rate `0 ≤ α < 1`, noise has standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param n: only used for sparse secrets
    :param c: explicit constant `c`

    EXAMPLE::

        sage: from estimator import _dual_scale_factor, alphaf
        sage: _dual_scale_factor(True, 8./2^15, 2^15)
        (1.00000000000000, 3.19153824321146)

        sage: _dual_scale_factor(alphaf(sqrt(2), 2^13, True), \
                                 alpha=alphaf(sqrt(16/3), 2^13, True), q=2^13)
        (1.63299316185545, 1.41421356237309)

        sage: _dual_scale_factor((-1,1), alpha=8./2^15, q=2^15)
        (3.90882009522336, 0.816496580927726)

        sage: _dual_scale_factor(((-1,1), 64), alpha=8./2^15, q=2^15, n=256)
        (6.38307648642292, 0.500000000000000)

        sage: _dual_scale_factor((-3,3), alpha=8./2^15, q=2^15)
        (1.59576912160573, 2.00000000000000)

        sage: _dual_scale_factor(((-3,3), 64), alpha=8./2^15, q=2^15, n=256)
        (2.95479025475795, 1.08012344973464)

        sage: _dual_scale_factor((-3,2), alpha=8./2^15, q=2^15)
        (1.86877344262086, 1.70782512765993)

        sage: _dual_scale_factor(((-3,2), 64), alpha=8./2^15, q=2^15, n=256)
        (3.31392819210159, 0.963068014212911)

    ..  note :: This function assumes that the bounds are of opposite sign, and that the
        distribution is centred around zero.
    """
    stddev = RR(stddevf(alpha * q))
    # Calculate scaling in the case of bounded_uniform secret distribution
    # NOTE: We assume a <= 0 <= b
    if SDis.is_small(secret_distribution, alpha=alpha):
        stddev_s = SDis.variance(secret_distribution, alpha=alpha, q=q, n=n).sqrt()
        if c is None:
            # |<v,s>| = |<w,e>| → c * \sqrt{n} * \sqrt{σ_{s_i}^2 + E(s_i)^2} == \sqrt{m} * σ
            # TODO: we are assuming n == m here! The coefficient formula for general m
            #       is the one below * sqrt(m/n)
            c = RR(stddev / stddev_s)
    else:
        stddev_s = stddev
        c = RR(1)

    stddev_s = RR(stddev_s)

    return c, stddev_s


def _dual(
    n, alpha, q, secret_distribution=True, m=oo, success_probability=0.99, reduction_cost_model=reduction_default_cost
):
    """
    Estimate cost of solving LWE using dual attack as described in [MicReg09]_

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param m: number of LWE samples `m > 0`
    :param success_probability: targeted success probability < 1
    :param reduction_cost_model: cost model for lattice reduction

    ..  [MicReg09] Micciancio, D., & Regev, O.  (2009).  Lattice-based cryptography.  In D.  J.
        Bernstein, J.  Buchmann, & E.  Dahmen (Eds.), Post-Quantum Cryptography (pp.  147–191).
        Berlin, Heidelberg, New York: Springer, Heidelberg.

    EXAMPLE::

        sage: from estimator import Param, BKZ, dual
        sage: n, alpha, q = Param.Regev(256)

        sage: dual(n, alpha, q)
            rop:  2^197.2
              m:      751
            red:  2^197.2
        delta_0: 1.005048
           beta:      282
              d:      751
            |v|: 1923.968
         repeat:   2^35.0
        epsilon:  2^-16.0

        sage: dual(n, alpha, q, secret_distribution=True, m=n)
            rop:  2^255.2
              m:      512
            red:  2^255.2
        delta_0: 1.004627
           beta:      322
              d:      512
            |v|:   2^11.4
         repeat:   2^67.0
        epsilon:  2^-32.0

        sage: dual(n, alpha, q, secret_distribution=False, m=2*n)
            rop:  2^255.2
              m:      512
            red:  2^255.2
        delta_0: 1.004627
           beta:      322
              d:      512
            |v|:   2^11.4
         repeat:   2^67.0
        epsilon:  2^-32.0

        sage: dual(n, alpha, q, reduction_cost_model=BKZ.sieve)
            rop:  2^142.9
              m:      787
            red:  2^142.9
        delta_0: 1.004595
           beta:      325
              d:      787
            |v|: 1360.451
         repeat:   2^19.0
        epsilon: 0.003906

    ..  note :: this is the standard dual attack, for the small secret variant see
    ``dual_scale``

    """

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)

    if m < 1:
        raise InsufficientSamplesError("m=%d < 1" % m)

    RR = parent(alpha)
    f = lambda eps: RR(sqrt(log(1 / eps) / pi))  # noqa

    if SDis.is_small(secret_distribution, alpha=alpha):
        m = m + n
    log_delta_0 = log(f(success_probability) / alpha, 2) ** 2 / (4 * n * log(q, 2))
    delta_0 = min(RR(2) ** log_delta_0, RR(1.02190))  # at most LLL
    m_optimal = lattice_reduction_opt_m(n, q, delta_0)
    if m > m_optimal:
        m = m_optimal
    else:
        log_delta_0 = log(f(success_probability) / alpha, 2) / m - RR(log(q, 2) * n) / (m ** 2)
        delta_0 = RR(2 ** log_delta_0)

    # check for valid delta
    # TODO delta_0f(m) is HKZ reduction, and thus it is identical 1, i.e. meaningless.
    # A better check here would be good.
    if delta_0 < delta_0f(m):
        raise OutOfBoundsError("delta_0 = %f < %f" % (delta_0, delta_0f(m)))

    ret = lattice_reduction_cost(reduction_cost_model, delta_0, m, B=log(q, 2))
    ret[u"m"] = m
    ret[u"d"] = m
    ret[u"|v|"] = RR(delta_0 ** m * q ** (n / m))
    return ret.reorder(["rop", "m"])


dual = partial(rinse_and_repeat, _dual, repeat_select={"m": False})


def dual_scale(
    n,
    alpha,
    q,
    secret_distribution,
    m=oo,
    success_probability=0.99,
    reduction_cost_model=reduction_default_cost,
    c=None,
    use_lll=True,
):
    """
    Estimate cost of solving LWE by finding small `(y,x/c)` such that `y ⋅ A ≡ c ⋅ x \bmod q` as
    described in [EC:Abrecht17]_

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param m: number of LWE samples `m > 0`
    :param success_probability: targeted success probability < 1
    :param reduction_cost_model: cost model for lattice reduction
    :param c: explicit constant `c`
    :param use_lll: use LLL calls to produce more small vectors

    EXAMPLES::

        sage: from estimator import Param, dual_scale

        sage: dual_scale(*Param.Regev(256), secret_distribution=(-1,1))
            rop:  2^101.8
              m:      288
            red:  2^101.8
        delta_0: 1.006648
           beta:      183
         repeat:   2^65.4
              d:      544
              c:   31.271

        sage: dual_scale(*Param.Regev(256), secret_distribution=(-1,1), m=200)
            rop:  2^106.0
              m:      200
            red:  2^106.0
        delta_0: 1.006443
           beta:      192
         repeat:   2^68.0
              d:      456
              c:   31.271

        sage: dual_scale(*Param.Regev(256), secret_distribution=((-1,1), 64))
            rop:   2^93.7
              m:      258
            red:   2^93.7
        delta_0: 1.006948
           beta:      170
         repeat:   2^56.0
              d:      514
              c:   51.065

    .. [Albrecht17] Albrecht, M.  R.  (2017).  On dual lattice attacks against small-secret LWE and
       parameter choices in helib and SEAL.  In J.  Coron, & J.  B.  Nielsen, EUROCRYPT} 2017, Part {II
       (pp.  103–129).
    """

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)
    RR = parent(alpha)

    # stddev of the error
    stddev = RR(stddevf(alpha * q))

    m_max = m

    if not SDis.is_small(secret_distribution, alpha=alpha):
        m_max -= n  # We transform to normal form, spending n samples

    c, stddev_s = _dual_scale_factor(secret_distribution, alpha=alpha, q=q, n=n, c=c)

    best = dual(n=n, alpha=alpha, q=q, m=m, reduction_cost_model=reduction_cost_model)
    delta_0 = best["delta_0"]

    if use_lll:
        rerand_scale = 2
    else:
        rerand_scale = 1

    while True:
        m_optimal = lattice_reduction_opt_m(n, q / c, delta_0)
        d = ZZ(min(m_optimal, m_max + n))
        m = d - n

        # the vector found will have norm v
        v = rerand_scale * delta_0 ** d * (q / c) ** (n / d)

        # each component has stddev v_
        v_ = v / RR(sqrt(d))

        # we split our vector in two parts.
        v_r = sigmaf(RR(stddev * sqrt(m) * v_))  # 1. v_r is multiplied with the error stddev (dimension m-n)
        v_l = sigmaf(RR(c * stddev_s * sqrt(n) * v_))  # 2. v_l is the rounding noise (dimension n)

        ret = lattice_reduction_cost(reduction_cost_model, delta_0, d, B=log(q, 2))

        repeat = max(amplify_sigma(success_probability, (v_r, v_l), q), RR(1))
        if use_lll:
            lll = BKZ.LLL(d, log(q, 2))
        else:
            lll = None
        ret = ret.repeat(times=repeat, lll=lll)

        ret[u"m"] = m
        ret[u"repeat"] = repeat
        ret[u"d"] = d
        ret[u"c"] = c

        ret = ret.reorder(["rop", "m"])
        logging.getLogger("repeat").debug(ret)

        if best is None:
            best = ret

        if ret["rop"] > best["rop"]:
            break

        best = ret
        delta_0 = delta_0 + RR(0.00005)

    return best


# Combinatorial


def mitm(n, alpha, q, secret_distribution=True, m=oo, success_probability=0.99):
    """
    Meet-in-the-Middle attack as described in [AlbPlaSco15]_

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param m: number of LWE samples `m > 0`
    :param success_probability: targeted success probability < 1

    .. [AlbPlaSco15] Albrecht, M. R., Player, R., & Scott, S. (2015). On the concrete hardness of
       Learning with Errors. Journal of Mathematical Cryptology, 9(3), 169–203.

    """
    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)
    ret = Cost()
    RR = alpha.parent()

    if not SDis.is_small(secret_distribution, alpha=alpha):
        m = m - n
        secret_distribution = True
        ret["m"] = n
    else:
        ret["m"] = 0

    if m < 1:
        raise InsufficientSamplesError("m=%d < 1" % m)

    t = ceil(2 * sqrt(log(n)))
    try:
        a, b = SDis.bounds(secret_distribution)
        size = b - a + 1
    except ValueError:
        size = (2 * t * alpha * q) + 1

    m_required = ceil((log((2 * n) / ((size ** (n / 2)) - 1))) / (log(size / q)))
    if m is not None and m < m_required:
        raise InsufficientSamplesError("m=%d < %d (required)" % (m, m_required))
    else:
        m = m_required

    if (3 * t * alpha) * m > 1 - 1 / (2 * n):
        raise ValueError("Cannot find m to satisfy constraints (noise too big).")

    ret["rop"] = RR(size ** (n / 2) * 2 * n * m)
    ret["mem"] = RR(size ** (n / 2) * m)
    ret["m"] += m

    return ret.reorder(["rop", "m", "mem"])


cfft = 1  # convolutions mod q


def _bkw_coded(n, alpha, q, secret_distribution=True, m=oo, success_probability=0.99, t2=0, b=2, ntest=None):  # noqa
    """
    Coded-BKW.

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param t2:                   number of coded BKW steps (≥ 0)
    :param b:                    table size (≥ 1)
    :param success_probability: targeted success probability < 1
    :param ntest:                optional parameter ntest

    """
    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability)
    sigma = stddevf(alpha * q)  # [C:GuoJohSta15] use σ = standard deviation
    RR = alpha.parent()

    cost = Cost()

    # Our cost is mainly determined by q**b, on the other hand there are
    # expressions in q**(l+1) below, hence, we set l = b - 1. This allows to
    # achieve the performance reported in [C:GuoJohSta15].

    b = ZZ(b)
    cost["b"] = b
    l = b - 1  # noqa
    cost["l"] = l

    try:
        secret_bounds = SDis.bounds(secret_distribution)
        d = secret_bounds[1] - secret_bounds[0] + 1
    except ValueError:
        d = 3 * sigma  # TODO make this dependent on success_probability

    # cost["d"] = d
    # cost[u"γ"] = gamma

    def N(i, sigma_set):
        """
        Return `N_i` for the `i`-th `[N_i, b]` linear code.

        :param i: index
        :param sigma_set: target noise level
        """
        return floor(b / (1 - log(12 * sigma_set ** 2 / ZZ(2) ** i, q) / 2))

    def find_ntest(n, l, t1, t2, b):  # noqa
        """
        If the parameter `ntest` is not provided, we use this function to estimate it.

        :param n: LWE dimension `n > 0`
        :param l:  table size for hypothesis testing
        :param t1: number of normal BKW steps
        :param t2: number of coded BKW steps
        :param b:  table size for BKW steps

        """

        # there is no hypothesis testing because we have enough normal BKW
        # tables to cover all of of n
        if t1 * b >= n:
            return 0

        # solve for nest by aiming for ntop == 0
        ntest = var("ntest")
        sigma_set = sqrt(q ** (2 * (1 - l / ntest)) / 12)
        ncod = sum([N(i, sigma_set) for i in range(1, t2 + 1)])
        ntop = n - ncod - ntest - t1 * b

        try:
            start = max(ZZ(round(find_root(ntop, 2, n - t1 * b + 1, rtol=0.1))) - 1, 2)
        except RuntimeError:
            start = 2
        ntest_min = 1
        for ntest in range(start, n - t1 * b + 1):
            if abs(ntop(ntest=ntest).n()) < abs(ntop(ntest=ntest_min).n()):
                ntest_min = ntest
            else:
                break
        return ZZ(ntest_min)

    # we compute t1 from N_i by observing that any N_i ≤ b gives no advantage
    # over vanilla BKW, but the estimates for coded BKW always assume
    # quantisation noise, which is too pessimistic for N_i ≤ b.
    t1 = 0
    if ntest is None:
        ntest_ = find_ntest(n, l, t1, t2, b)
    else:
        ntest_ = ntest
    sigma_set = sqrt(q ** (2 * (1 - l / ntest_)) / 12)
    Ni = [N(i, sigma_set) for i in range(1, t2 + 1)]
    t1 = len([e for e in Ni if e <= b])

    # there is no point in having more tables than needed to cover n
    if b * t1 > n:
        t1 = n // b
    t2 -= t1

    cost["t1"] = t1
    cost["t2"] = t2

    # compute ntest with the t1 just computed
    if ntest is None:
        ntest = find_ntest(n, l, t1, t2, b)

    # if there's no ntest then there's no `σ_{set}` and hence no ncod
    if ntest:
        sigma_set = sqrt(q ** (2 * (1 - l / ntest)) / 12)
        # cost[u"σ_set"] = RR(sigma_set)
        ncod = sum([N(i, sigma_set) for i in range(1, t2 + 1)])
    else:
        ncod = 0

    ntot = ncod + ntest
    ntop = max(n - ncod - ntest - t1 * b, 0)
    cost["ncod"] = ncod  # coding step
    cost["ntop"] = ntop  # guessing step, typically zero
    cost["ntest"] = ntest  # hypothesis testing

    # Theorem 1: quantization noise + addition noise
    s_var = SDis.variance(secret_distribution, alpha, q, n=n)
    coding_variance = s_var * sigma_set ** 2 * ntot
    sigma_final = RR(sqrt(2 ** (t1 + t2) * sigma ** 2 + coding_variance))
    # cost[u"σ_final"] = RR(sigma_final)

    M = amplify_sigma(success_probability, sigmaf(sigma_final), q)
    if M is oo:
        cost["rop"] = oo
        cost["m"] = oo
        return cost
    m = (t1 + t2) * (q ** b - 1) / 2 + M
    cost["m"] = RR(m)

    if not SDis.is_small(secret_distribution, alpha=alpha):
        # Equation (7)
        n_ = n - t1 * b
        C0 = (m - n_) * (n + 1) * ceil(n_ / (b - 1))
        assert C0 >= 0
        # cost["C0(gauss)"] = RR(C0)
    else:
        C0 = 0

    # Equation ( 8)
    C1 = sum([(n + 1 - i * b) * (m - i * (q ** b - 1) / 2) for i in range(1, t1 + 1)])
    assert C1 >= 0
    # cost["C1(bkw)"] = RR(C1)

    # Equation (9)
    C2_ = sum([4 * (M + i * (q ** b - 1) / 2) * N(i, sigma_set) for i in range(1, t2 + 1)])
    C2 = RR(C2_)
    for i in range(1, t2 + 1):
        C2 += RR(ntop + ntest + sum([N(j, sigma_set) for j in range(1, i + 1)])) * (M + (i - 1) * (q ** b - 1) / 2)
    assert C2 >= 0
    # cost["C2(coded)"] = RR(C2)

    # Equation (10)
    C3 = M * ntop * (2 * d + 1) ** ntop
    assert C3 >= 0
    # cost["C3(guess)"] = RR(C3)

    # Equation (11)
    C4_ = 4 * M * ntest
    C4 = C4_ + (2 * d + 1) ** ntop * (cfft * q ** (l + 1) * (l + 1) * log(q, 2) + q ** (l + 1))
    assert C4 >= 0
    # cost["C4(test)"] = RR(C4)

    C = (C0 + C1 + C2 + C3 + C4) / (erf(d / sqrt(2 * sigma)) ** ntop)  # TODO don't ignore success probability
    cost["rop"] = RR(C)
    cost["mem"] = (t1 + t2) * q ** b

    cost = cost.reorder(["rop", "m", "mem", "b", "t1", "t2"])

    return cost


def bkw_coded(n, alpha, q, secret_distribution=True, m=oo, success_probability=0.99, ntest=None):
    """
    Coded-BKW as described in [C:GuoJohSta15]

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param success_probability: targeted success probability < 1
    :param samples: the number of available samples

    EXAMPLE::

        sage: from estimator import Param, bkw_coded
        sage: n, alpha, q = Param.Regev(64)
        sage: bkw_coded(n, alpha, q)
             rop:   2^48.8
               m:   2^39.6
             mem:   2^39.6
               b:        3
              t1:        2
              t2:       10
               l:        2
            ncod:       53
            ntop:        0
           ntest:        6

    .. [GuoJohSta15] Guo, Q., Johansson, T., & Stankovski, P.  (2015).  Coded-BKW: solving LWE using lattice
       codes.  In R.  Gennaro, & M.  J.  B.  Robshaw, CRYPTO~2015, Part~I (pp.  23–42).  :
       Springer, Heidelberg.

    """
    bstart = ceil(log(q, 2))

    def _run(b=2):
        # the noise is 2**(t1+t2) * something so there is no need to go beyond, say, q**2
        r = binary_search(
            _bkw_coded,
            2,
            min(n // b, ceil(2 * log(q, 2))),
            "t2",
            predicate=lambda x, best: x["rop"] <= best["rop"] and (best["m"] > m or x["m"] <= m),
            b=b,
            n=n,
            alpha=alpha,
            q=q,
            secret_distribution=secret_distribution,
            success_probability=success_probability,
            m=m,
            ntest=ntest,
        )
        return r

    best = binary_search(
        _run, 2, 3 * bstart, "b", predicate=lambda x, best: x["rop"] <= best["rop"] and (best["m"] > m or x["m"] <= m)
    )
    # binary search cannot fail. It just outputs some X with X["oracle"]>m.
    if best["m"] > m:
        raise InsufficientSamplesError("m=%d < %d (required)" % (m, best["m"]))
    return best


# Algebraic


@cached_function
def have_magma():
    try:
        magma(1)
        return True
    except TypeError:
        return False


def gb_cost(n, D, omega=2, prec=None):
    """
    Estimate the complexity of computing a Gröbner basis.

    :param n: number of variables `n > 0`
    :param D: tuple of `(d,m)` pairs where `m` is number polynomials and `d` is a degree
    :param omega: linear algebra exponent, i.e. matrix-multiplication costs `O(n^ω)` operations.
    :param prec: compute power series up to this precision (default: `2n`)

    """
    prec = 2*n if prec is None else prec

    if have_magma():
        R = magma.PowerSeriesRing(QQ, prec)
        z = R.gen(1)
        coeff = lambda f, d: f.Coefficient(d)  # noqa
        s = 1
    else:
        R = PowerSeriesRing(QQ, "z", prec)
        z = R.gen()
        z = z.add_bigoh(prec)
        coeff = lambda f, d: f[d]  # noqa
        s = R(1)
        s = s.add_bigoh(prec)

    for d, m in D:
        s *= (1 - z ** d) ** m
    s /= (1 - z) ** n

    retval = Cost([("rop", oo), ("Dreg", oo)])

    for dreg in range(prec):
        if coeff(s, dreg) < 0:
            break
    else:
        return retval

    retval["Dreg"] = dreg
    retval["rop"] = binomial(n + dreg, dreg) ** omega
    retval["mem"] = binomial(n + dreg, dreg) ** 2

    return retval


def arora_gb(n, alpha, q, secret_distribution=True, m=oo, success_probability=0.99, omega=2):
    """
    Arora-GB as described in [AroGe11,ACFP14]_

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param m: number of LWE samples `m > 0`
    :param success_probability: targeted success probability < 1
    :param omega: linear algebra constant

    ..  [ACFP14] Albrecht, M.  R., Cid, C., Jean-Charles Faugère, & Perret, L.  (2014).
        Algebraic algorithms for LWE.

    ..  [AroGe11] Arora, S., & Ge, R.  (2011).  New algorithms for learning in presence of
        errors.  In L.  Aceto, M.  Henzinger, & J.  Sgall, ICALP 2011, Part~I (pp.  403–415).  :
        Springer, Heidelberg.
    """

    n, alpha, q, success_probability = Param.preprocess(n, alpha, q, success_probability, prec=2 * log(n, 2) * n)

    RR = alpha.parent()
    stddev = RR(stddevf(RR(alpha) * q))

    if stddev >= 1.1 * sqrt(n):
        return {"rop": oo}

    if SDis.is_small(secret_distribution, alpha=alpha):
        try:
            a, b = SDis.bounds(secret_distribution)
            d2 = b - a + 1
        except ValueError:
            d2 = 2 * ceil(3 * stddev) + 1
        d2 = [(d2, n)]
    else:
        d2 = []

    ps_single = lambda C: RR(1 - (2 / (C * RR(sqrt(2 * pi))) * exp(-(C ** 2) / 2)))  # noqa

    m_req = floor(exp(RR(0.75) * n))
    d = n
    t = ZZ(floor((d - 1) / 2))
    C = t / stddev
    pred = gb_cost(n, [(d, m_req)] + d2, omega)
    pred["t"] = t
    pred["m"] = m_req
    pred = pred.reorder(["rop", "m", "Dreg", "t"])

    t = ceil(t / 3)
    best = None
    stuck = 0
    for t in srange(t, n):
        d = 2 * t + 1
        C = RR(t / stddev)
        if C < 1:  # if C is too small, we ignore it
            continue
        # Pr[success]^m = Pr[overall success]
        m_req = log(success_probability, 2) / log(ps_single(C), 2)
        if m_req < n:
            continue
        m_req = floor(m_req)

        current = gb_cost(n, [(d, m_req)] + d2, omega)

        if current["Dreg"] is None:
            continue

        current["t"] = t
        current["m"] = m_req

        current = current.reorder(["rop", "m", "Dreg", "t"])

        logging.getLogger("repeat").debug(current)

        if best is None:
            best = current
        else:
            if best["rop"] > current["rop"] and current["m"] <= m:
                best = current
                stuck = 0
            else:
                stuck += 1
                if stuck >= 5:
                    break
    return best


# Toplevel function


def estimate_lwe(  # noqa
    n,
    alpha=None,
    q=None,
    secret_distribution=True,
    m=oo,
    reduction_cost_model=reduction_default_cost,
    skip=("mitm", "arora-gb", "bkw"),
):
    """
    Highlevel-function for estimating security of LWE parameter sets

    :param n: LWE dimension `n > 0`
    :param alpha: noise rate `0 ≤ α < 1`, noise will have standard deviation `αq/\\sqrt{2π}`
    :param q: modulus `0 < q`
    :param m: number of LWE samples `m > 0`
    :param secret_distribution: distribution of secret, see module level documentation for details
    :param reduction_cost_model: use this cost model for lattice reduction
    :param skip: skip these algorithms

    EXAMPLE::

        sage: from estimator import estimate_lwe, Param, BKZ, alphaf
        sage: d = estimate_lwe(*Param.Regev(128))
        usvp: rop:  ≈2^57.3,  red:  ≈2^57.3,  δ_0: 1.009214,  β:  101,  d:  349,  m:      220
         dec: rop:  ≈2^61.9,  m:      229,  red:  ≈2^61.9,  δ_0: 1.009595,  β:   93,  d:  357,  ...
        dual: rop:  ≈2^81.1,  m:      380,  red:  ≈2^81.1,  δ_0: 1.008631,  β:  115,  d:  380,  ...

        sage: d = estimate_lwe(**Param.LindnerPeikert(256, dict=True))
        usvp: rop: ≈2^127.8,  red: ≈2^127.8,  δ_0: 1.005788,  β:  228,  d:  588,  m:      331
         dec: rop: ≈2^137.3,  m:      337,  red: ≈2^137.3,  δ_0: 1.005976,  β:  217,  d:  593,  ...
        dual: rop: ≈2^166.0,  m:      368,  red: ≈2^166.0,  δ_0: 1.005479,  β:  249,  repeat: ...

        sage: d = estimate_lwe(*Param.LindnerPeikert(256), secret_distribution=(-1,1))
        Warning: the LWE secret is assumed to have Hamming weight 171.
        usvp: rop:  ≈2^98.0,  red:  ≈2^98.0,  δ_0: 1.006744,  β:  178,  d:  499,  m:      242,  ...
         dec: rop: ≈2^137.3,  m:      337,  red: ≈2^137.3,  δ_0: 1.005976,  β:  217,  d:  593,  ...
        dual: rop: ≈2^108.4,  m:      270,  red: ≈2^108.3,  δ_0: 1.006390,  β:  195,  repeat:  ...

        sage: d = estimate_lwe(*Param.LindnerPeikert(256), secret_distribution=(-1,1), reduction_cost_model=BKZ.sieve)
        Warning: the LWE secret is assumed to have Hamming weight 171.
        usvp: rop:  ≈2^80.3,  red:  ≈2^80.3,  δ_0: 1.006744,  β:  178,  d:  499,  m:      242,  ...
         dec: rop: ≈2^111.8,  m:      369,  red: ≈2^111.8,  δ_0: 1.005423,  β:  253,  d:  625,  ...
        dual: rop:  ≈2^90.6,  m:      284,  red:  ≈2^90.6,  δ_0: 1.006065,  β:  212,  ...

        sage: d = estimate_lwe(n=100, alpha=8/2^20, q=2^20, skip="arora-gb")
        mitm: rop: ≈2^329.2,  m:       23,  mem: ≈2^321.5
        usvp: rop:  ≈2^32.4,  red:  ≈2^32.4,  δ_0: 1.013310,  β:   40,  d:  141,  m:       40
         dec: rop:  ≈2^34.0,  m:      156,  red:  ≈2^34.0,  δ_0: 1.021398,  β:   40,  d:  256,  ...
        dual: rop:  ≈2^35.5,  m:      311,  red:  ≈2^35.5,  δ_0: 1.014423,  β:   40,  d:  311,  ...
         bkw: rop:  ≈2^53.6,  m:  ≈2^43.5,  mem:  ≈2^44.5,  b:   2,  t1:   5,  t2:  18,  l:   1,  ...

        sage: d = estimate_lwe(n=100, secret_distribution=alphaf(1, 2^20, True), alpha=8/2^20, q=2^20)
        usvp: rop:  ≈2^32.2,  red:  ≈2^32.2,  δ_0: 1.013310,  β:   40,  d:  127,  m:       26
         dec: rop:  ≈2^34.0,  m:      156,  red:  ≈2^34.0,  δ_0: 1.021398,  β:   40,  d:  ...
        dual: rop:  ≈2^35.5,  m:      311,  red:  ≈2^35.5,  δ_0: 1.014423,  β:   40,  d:  ...


    """

    algorithms = OrderedDict()

    if skip is None:
        skip = ()
    try:
        skip = [s.strip().lower() for s in skip.split(",")]
    except AttributeError:
        pass
    skip = [s.strip().lower() for s in skip]

    if "mitm" not in skip:
        algorithms["mitm"] = mitm

    if "usvp" not in skip:
        if SDis.is_bounded_uniform(secret_distribution):
            algorithms["usvp"] = partial(
                drop_and_solve,
                primal_usvp,
                reduction_cost_model=reduction_cost_model,
                postprocess=False,
                decision=False,
            )
        else:
            algorithms["usvp"] = partial(primal_usvp, reduction_cost_model=reduction_cost_model)

    if "dec" not in skip:
        if SDis.is_sparse(secret_distribution) and SDis.is_ternary(secret_distribution):
            algorithms["dec"] = partial(
                drop_and_solve,
                primal_decode,
                reduction_cost_model=reduction_cost_model,
                postprocess=False,
                decision=False,
            )
        else:
            algorithms["dec"] = partial(primal_decode, reduction_cost_model=reduction_cost_model)

    if "dual" not in skip:
        if SDis.is_bounded_uniform(secret_distribution):
            algorithms["dual"] = partial(
                drop_and_solve, dual_scale, reduction_cost_model=reduction_cost_model, postprocess=True
            )
        elif SDis.is_small(secret_distribution, alpha=alpha):
            algorithms["dual"] = partial(dual_scale, reduction_cost_model=reduction_cost_model)
        else:
            algorithms["dual"] = partial(dual, reduction_cost_model=reduction_cost_model)

    if "bkw" not in skip:
        algorithms["bkw"] = bkw_coded

    if "arora-gb" not in skip:
        if SDis.is_sparse(secret_distribution) and SDis.is_small(secret_distribution, alpha=alpha):
            algorithms["arora-gb"] = partial(drop_and_solve, arora_gb)
        elif SDis.is_small(secret_distribution, alpha=alpha):
            algorithms["arora-gb"] = partial(switch_modulus, arora_gb)
        else:
            algorithms["arora-gb"] = arora_gb

    alg_width = max(len(key) for key in set(algorithms).difference(skip))
    cost_kwds = {"compact": False}

    logger = logging.getLogger("estimator")

    if SDis.is_bounded_uniform(secret_distribution) and not SDis.is_sparse(secret_distribution):
        h = SDis.nonzero(secret_distribution, n)
        logger.info("Warning: the LWE secret is assumed to have Hamming weight %s." % h)

    results = OrderedDict()
    for alg in algorithms:
        algf = algorithms[alg]
        try:
            tmp = algf(n, alpha, q, secret_distribution=secret_distribution, m=m)
            results[alg] = tmp
            logger.info("%s: %s" % (("%%%ds" % alg_width) % alg, results[alg].str(**cost_kwds)))
        except InsufficientSamplesError as e:
            logger.info(
                "%s: %s" % (("%%%ds" % alg_width) % alg, "insufficient samples to run this algorithm: '%s'" % str(e))
            )

    return results
