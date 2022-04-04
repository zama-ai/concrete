# -*- coding: utf-8 -*-

from dataclasses import dataclass

from sage.all import parent, RR, RealField, sqrt, pi, oo, ceil, binomial, exp


def stddevf(sigma):
    """
    Gaussian width parameter σ → standard deviation.

    :param sigma: Gaussian width parameter σ

    EXAMPLE::

        >>> from estimator.nd import stddevf
        >>> stddevf(64.0)
        25.532...

        >>> stddevf(64)
        25.532...

        >>> stddevf(RealField(256)(64)).prec()
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
    Standard deviation → Gaussian width parameter σ.

    :param stddev: standard deviation

    EXAMPLE::

        >>> from estimator.nd import stddevf, sigmaf
        >>> n = 64.0
        >>> sigmaf(stddevf(n))
        64.000...

        >>> sigmaf(RealField(128)(1.0))
        2.5066282746310005024157652848110452530
        >>> sigmaf(1.0)
        2.506628274631...
        >>> sigmaf(1)
        2.506628274631...

    """
    RR = parent(stddev)
    #  check that we got ourselves a real number type
    try:
        if abs(RR(0.5) - 0.5) > 0.001:
            RR = RealField(53)  # hardcode something
    except TypeError:
        RR = RealField(53)  # hardcode something
    return RR(sqrt(2 * pi)) * stddev


@dataclass
class NoiseDistribution:
    """
    All noise distributions are instances of this class.

    """

    # cut-off for Gaussian distributions
    gaussian_tail_bound = 2

    # probability that a coefficient falls within the cut-off
    gaussian_tail_prob = 1 - 2 * exp(-4 * pi)

    stddev: float
    mean: float = 0
    n: int = None
    bounds: tuple = None
    density: float = 1.0  # Hamming weight / dimension.
    tag: str = ""

    def __lt__(self, other):
        """
        We compare distributions by comparing their standard deviation.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.DiscreteGaussian(2.0) < ND.CenteredBinomial(18)
            True
            >>> ND.DiscreteGaussian(3.0) < ND.CenteredBinomial(18)
            False
            >>> ND.DiscreteGaussian(4.0) < ND.CenteredBinomial(18)
            False

        """
        try:
            return self.stddev < other.stddev
        except AttributeError:
            return self.stddev < other

    def __le__(self, other):
        """
        We compare distributions by comparing their standard deviation.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.DiscreteGaussian(2.0) <= ND.CenteredBinomial(18)
            True
            >>> ND.DiscreteGaussian(3.0) <= ND.CenteredBinomial(18)
            True
            >>> ND.DiscreteGaussian(4.0) <= ND.CenteredBinomial(18)
            False

        """
        try:
            return self.stddev <= other.stddev
        except AttributeError:
            return self.stddev <= other

    def __str__(self):
        """
        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.DiscreteGaussianAlpha(0.01, 7681)
            D(σ=30.64)

        """
        if self.n:
            return f"D(σ={float(self.stddev):.2f}, μ={float(self.mean):.2f}, n={int(self.n)})"
        else:
            return f"D(σ={float(self.stddev):.2f}, μ={float(self.mean):.2f})"

    def __repr__(self):
        if self.mean == 0.0:
            return f"D(σ={float(self.stddev):.2f})"
        else:
            return f"D(σ={float(self.stddev):.2f}, μ={float(self.mean):.2f})"

    def __hash__(self):
        """
        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> hash(ND(3.0, 1.0)) == hash((3.0, 1.0, None))
            True

        """
        return hash((self.stddev, self.mean, self.n))

    def __len__(self):
        """
        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> D = ND.SparseTernary(1024, p=128, m=128)
            >>> len(D)
            1024
            >>> int(round(len(D) * float(D.density)))
            256

        """
        if self.n is not None:
            return self.n
        else:
            raise ValueError("Distribution has no length.")

    @property
    def is_Gaussian_like(self):
        return ("Gaussian" in self.tag) or ("CenteredBinomial" in self.tag)

    @property
    def is_bounded(self):
        return (self.bounds[1] - self.bounds[0]) < oo

    @property
    def is_sparse(self):
        """
        We consider a distribution "sparse" if its density is < 1/2.
        """
        # NOTE: somewhat arbitrary
        return self.density < 0.5

    def support_size(self, n=None, fraction=1.0):
        """
        Compute the size of the support covering the probability given as fraction.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> D = ND.Uniform(-3,3, 64)
            >>> D.support_size(fraction=.99)
            1207562882759477428726191443614714994252339953407098880
            >>> D = ND.SparseTernary(64, 8)
            >>> D.support_size()
            32016101348447354880
        """
        if not n:
            if not self.n:
                raise ValueError(f"Length required to determine support size, but n was {n}.")
            n = self.n

        if "SparseTernary" in self.tag:
            h = self.h
            # TODO: this is assuming that the non-zero entries are uniform over {-1,1}
            # need p and m for more accurate calculation
            size = 2 ** h * binomial(n, h) * RR(fraction)
        elif self.is_bounded:
            # TODO: this might be suboptimal/inaccurate for binomial distribution
            a, b = self.bounds
            size = RR(fraction) * (b - a + 1) ** n
        else:
            # Looks like nd is Gaussian
            # -> we'll treat it as bounded (with failure probability)
            t = self.gaussian_tail_bound
            p = self.gaussian_tail_prob

            if p ** n < fraction:
                raise NotImplementedError(
                    f"TODO(nd.support-size): raise t. {RR(p ** n)}, {n}, {fraction}"
                )

            b = 2 * t * sigmaf(self.stddev) + 1
            return (2 * b + 1) ** n
        return ceil(size)

    def get_hamming_weight(self, n=None):
        if hasattr(self, "h"):
            return self.h

        if not n:
            if not self.n:
                raise ValueError("Length required to determine hamming weight.")
            n = self.n
        return round(n * self.density)

    @staticmethod
    def DiscreteGaussian(stddev, mean=0, n=None):
        """
        A discrete Gaussian distribution with standard deviation ``stddev`` per component.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.DiscreteGaussian(3.0, 1.0)
            D(σ=3.00, μ=1.00)

        """
        return NoiseDistribution(
            stddev=RR(stddev), mean=RR(mean), n=n, bounds=(-oo, oo), tag="DiscreteGaussian"
        )

    @staticmethod
    def DiscreteGaussianAlpha(alpha, q, mean=0, n=None):
        """
        A discrete Gaussian distribution with standard deviation α⋅q/√(2π) per component.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.DiscreteGaussianAlpha(0.001, 2048)
            D(σ=0.82)

        """
        stddev = stddevf(alpha * q)
        return NoiseDistribution.DiscreteGaussian(stddev=RR(stddev), mean=RR(mean), n=n)

    @staticmethod
    def CenteredBinomial(eta, n=None):
        """
        Sample a_1, …, a_η, b_1, …, b_η and return Σ(a_i - b_i).

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.CenteredBinomial(8)
            D(σ=2.00)

        """
        stddev = sqrt(eta / 2.0)
        # TODO: density
        return NoiseDistribution(
            stddev=RR(stddev), mean=RR(0), n=n, bounds=(-eta, eta), tag="CenteredBinomial"
        )

    @staticmethod
    def Uniform(a, b, n=None):
        """
        Uniform distribution ∈ ``[a,b]``, endpoints inclusive.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.Uniform(-3, 3)
            D(σ=2.00)
            >>> ND.Uniform(-4, 3)
            D(σ=2.29, μ=-0.50)

        """
        if b < a:
            raise ValueError(f"upper limit must be larger than lower limit but got: {b} < {a}")
        m = b - a + 1
        mean = (a + b) / RR(2)
        stddev = sqrt((m ** 2 - 1) / RR(12))

        if a <= 0 and 0 <= b:
            density = 1.0 - 1.0 / m
        else:
            density = 0.0

        return NoiseDistribution(
            n=n, stddev=stddev, mean=mean, bounds=(a, b), density=density, tag="Uniform"
        )

    @staticmethod
    def UniformMod(q, n=None):
        """
        Uniform mod ``q``, with balanced representation.

        EXAMPLE::

            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.UniformMod(7)
            D(σ=2.00)
            >>> ND.UniformMod(8)
            D(σ=2.29, μ=-0.50)


        """
        a = -(q // 2)
        b = q // 2
        if q % 2 == 0:
            b -= 1
        return NoiseDistribution.Uniform(a, b, n=n)

    @staticmethod
    def SparseTernary(n, p, m=None):
        """
        Distribution of vectors of length ``n`` with ``p`` entries of 1 and ``m`` entries of -1, rest 0.

        EXAMPLE::
            >>> from estimator.nd import NoiseDistribution as ND
            >>> ND.SparseTernary(100, p=10)
            D(σ=0.45)
            >>> ND.SparseTernary(100, p=10, m=10)
            D(σ=0.45)
            >>> ND.SparseTernary(100, p=10, m=8)
            D(σ=0.42, μ=0.02)

        """
        if m is None:
            m = p

        if n == 0:
            # this might happen in the dual attack
            return NoiseDistribution(stddev=0, mean=0, density=0, bounds=(-1, 1), tag="SparseTernary", n=0)
        mean = RR(p / n - m / n)
        stddev = RR(sqrt((p + m) / n))
        density = RR((p + m) / n)
        D = NoiseDistribution(
            stddev=stddev, mean=mean, density=density, bounds=(-1, 1), tag="SparseTernary", n=n
        )
        D.h = p + m
        return D
