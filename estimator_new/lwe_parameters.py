# -*- coding: utf-8 -*-
from sage.all import oo, binomial, log, sqrt, ceil
from dataclasses import dataclass
from copy import copy
from .nd import NoiseDistribution
from .errors import InsufficientSamplesError


@dataclass
class LWEParameters:
    n: int
    q: int
    Xs: NoiseDistribution
    Xe: NoiseDistribution
    m: int = oo
    tag: str = None

    def __post_init__(self, **kwds):
        self.Xs = copy(self.Xs)
        self.Xs.n = self.n
        if self.m < oo:
            self.Xe = copy(self.Xe)
            self.Xe.n = self.m

    def normalize(self):
        """
        EXAMPLES:

        We perform the normal form transformation if χ_e < χ_s and we got the samples::

            >>> from estimator import *
            >>> Xs=ND.DiscreteGaussian(2.0)
            >>> Xe=ND.DiscreteGaussian(1.58)
            >>> LWE.Parameters(n=512, q=8192, Xs=Xs, Xe=Xe).normalize()
            LWEParameters(n=512, q=8192, Xs=D(σ=1.58), Xe=D(σ=1.58), m=+Infinity, tag=None)

        If m = n, we swap the secret and the noise::

            >>> from estimator import *
            >>> Xs=ND.DiscreteGaussian(2.0)
            >>> Xe=ND.DiscreteGaussian(1.58)
            >>> LWE.Parameters(n=512, q=8192, Xs=Xs, Xe=Xe, m=512).normalize()
            LWEParameters(n=512, q=8192, Xs=D(σ=1.58), Xe=D(σ=2.00), m=512, tag=None)

        """
        if self.m < 1:
            raise InsufficientSamplesError(f"m={self.m} < 1")

        # Normal form transformation
        if self.Xe < self.Xs and self.m >= 2 * self.n:
            return LWEParameters(
                n=self.n, q=self.q, Xs=self.Xe, Xe=self.Xe, m=self.m - self.n, tag=self.tag
            )
        # swap secret and noise
        # TODO: this is somewhat arbitrary
        if self.Xe < self.Xs and self.m < 2 * self.n:
            return LWEParameters(n=self.n, q=self.q, Xs=self.Xe, Xe=self.Xs, m=self.n, tag=self.tag)

        # nothing to do
        return self

    def updated(self, **kwds):
        """
        Return a new set of parameters updated according to ``kwds``.

        :param kwds: We set ``key`` to ``value`` in the new set of parameters.

        EXAMPLE::

            >>> from estimator import *
            >>> Kyber512
            LWEParameters(n=512, q=3329, Xs=D(σ=1.22), Xe=D(σ=1.22), m=512, tag='Kyber 512')
            >>> Kyber512.updated(m=1337)
            LWEParameters(n=512, q=3329, Xs=D(σ=1.22), Xe=D(σ=1.22), m=1337, tag='Kyber 512')

        """
        d = dict(self.__dict__)
        d.update(kwds)
        return LWEParameters(**d)

    def amplify_m(self, m):
        """
        Return a LWE instance parameters with ``m`` samples produced from the samples in this instance.

        :param m: New number of samples.

        EXAMPLE::

            >>> from sage.all import binomial, log
            >>> from estimator import *
            >>> Kyber512
            LWEParameters(n=512, q=3329, Xs=D(σ=1.22), Xe=D(σ=1.22), m=512, tag='Kyber 512')
            >>> Kyber512.amplify_m(2**100)
            LWEParameters(n=512, q=3329, Xs=D(σ=1.22), Xe=D(σ=4.58), m=..., tag='Kyber 512')

        We can produce 2^100 samples by random ± linear combinations of 12 vectors::

            >>> float(sqrt(12.)), float(log(binomial(1024, 12) , 2.0)) + 12
            (3.46..., 103.07...)

        """
        if m <= self.m:
            return self
        if self.m == oo:
            return self
        d = dict(self.__dict__)

        if self.Xe.mean != 0:
            raise NotImplementedError("Amplifying for μ≠0 not implemented.")

        for k in range(ceil(log(m, 2.0))):
            # - binom(n,k) positions
            #  -two signs per position (+1,-1)
            # - all "-" and all "+" are the same
            if binomial(self.m, k) * 2 ** k - 1 >= m:
                Xe = NoiseDistribution.DiscreteGaussian(float(sqrt(k) * self.Xe.stddev))
                d["Xe"] = Xe
                d["m"] = ceil(m)
                return LWEParameters(**d)
        else:
            raise NotImplementedError(
                f"Cannot amplify to ≈2^{log(m,2):1} using {{+1,-1}} additions."
            )

    def switch_modulus(self):
        """
        Apply modulus switching and return new instance.

        See [JMC:AlbPlaSco15]_ for details.

        EXAMPLE::

            >>> from estimator import *
            >>> LWE.Parameters(n=128, q=7681, Xs=ND.UniformMod(3), Xe=ND.UniformMod(11)).switch_modulus()
            LWEParameters(n=128, q=5289, Xs=D(σ=0.82), Xe=D(σ=3.08), m=+Infinity, tag=None)

        """
        n = self.Xs.density * len(self.Xs)

        # n uniform in -(0.5,0.5) ± stddev(χ_s)
        Xr_stddev = sqrt(n / 12) * self.Xs.stddev  # rounding noise
        # χ_r == p/q ⋅ χ_e # we want the rounding noise match the scaled noise
        p = ceil(Xr_stddev * self.q / self.Xe.stddev)

        scale = float(p) / self.q

        # there is no point in scaling if the improvement is eaten up by rounding noise
        if scale > 1 / sqrt(2):
            return self

        return LWEParameters(
            self.n,
            p,
            Xs=self.Xs,
            Xe=NoiseDistribution.DiscreteGaussian(sqrt(2) * self.Xe.stddev * scale),
            m=self.m,
            tag=self.tag + ",scaled" if self.tag else None,
        )

    def __hash__(self):
        return hash((self.n, self.q, self.Xs, self.Xe, self.m, self.tag))
