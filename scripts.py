import matplotlib.pyplot as plt
import numpy as np
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionIntegerSampler
from concrete_params import concrete_LWE_params, concrete_RLWE_params
# easier to just load the estimator
load("estimator.py")

# define the four cost models used for Concrete (2 classical, 2 quantum)
# note that classical and quantum are the two models used in the "HE Std"

classical = lambda beta, d, B: ZZ(2) ** RR(0.292 * beta + 16.4 + log(8 * d, 2)),
quantum = lambda beta, d, B: ZZ(2) ** RR(0.265 * beta + 16.4 + log(8 * d, 2)),
classical_conservative = lambda beta, d, B: ZZ(2) ** RR(0.292 * beta),
quantum_conservative = lambda beta, d, B: ZZ(2) ** RR(0.265 * beta),
cost_models = [classical, quantum, classical_conservative, quantum_conservative, BKZ.enum]

# functions to automate parameter selection

def get_security_level(estimate, decimal_places = 2):
    """ Function to get the security level from an LWE Estimator output, 
    i.e. returns only the bit-security level (without the attack params)
    :param estimate: the input estimate
    :param decimal_places: the number of decimal places"%.2f" %

    EXAMPLE:
    sage: x = estimate_lwe(n = 256, q = 2**32, alpha = RR(8/2**32))
    sage: get_security_level(x)
    33.8016789754458
    """

    levels = []

    # use try/except to cover cases where we only consider one or two attacks

    try:
        levels.append(estimate["usvp"]["rop"])

    except:
        pass

    try:
        levels.append(estimate["dec"]["rop"])

    except:
        pass

    try:
        levels.append(estimate["dual"]["rop"])

    except:
        pass

    # take the minimum attack cost (in bits)
    security_level = round(log(min(levels), 2), decimal_places)

    return security_level


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

        for model in cost_models:
            try:
                model = model[0]
            except:
                model = model
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                        reduction_cost_model=model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
            results.append(get_security_level(estimate))

        RESULTS.append(results)

    return RESULTS

def latexit(results):
    """
    A function which takes the output of get_all_security_levels() and
    turns it into a latex table
    :param results: the security levels

    sage: X = get_all_security_levels(concrete_LWE_params)
    sage: latextit(X)
    \begin{tabular}{llllll}
    LWE128_256 & $126.69$ & $117.57$ & $98.7$ & $89.57$ & $217.55$ \\
    LWE128_512 & $135.77$ & $125.92$ & $106.58$ & $96.73$ & $218.53$ \\
    LWE128_638 & $135.27$ & $125.49$ & $105.7$ & $95.93$ & $216.81$ \\
    [...]
    """

    table(results)

    return latex(table(results))

def inequality(x, y):
    """ A function which compresses the conditions
    x < y and x > y into a single condition via a 
    multiplier
    """
    if x <= y:
        return 1

    if x > y:
        return -1


def automated_param_select_n(sd, n=None, q=2 ** 32, reduction_cost_model=BKZ.sieve, secret_distribution=(0, 1),
                             target_security=128):
    """ A function used to generate the smallest value of n which allows for 
    target_security bits of security, for the input values of (sd,q)
    :param sd: the standard deviation of the error
    :param q: the LWE modulus (q = 2**32, 2**64 in TFHE)
    :param reduction_cost_model: the BKZ cost model considered, BKZ.sieve is default
    :param secret_distribution: the LWE secret distribution
    :param target_security: the target number of bits of security, 128 is default

    EXAMPLE:
    sage: X = automated_param_select_n(sd = -25, q = 2**32)
    sage: X
    1054
    """

    if n is None:
        # pick some random n which gets us close (based on concrete_LWE_params)
        n = sd * (-25) * (target_security/80)

    sd = 2 ** sd * q
    alpha = sqrt(2 * pi) * sd / RR(q)


    # initial estimate, to determine if we are above or below the target security level
    estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
    security_level = get_security_level(estimate)
    z = inequality(security_level, target_security)

    while z * security_level < z * target_security:
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
        security_level = get_security_level(estimate)

        n += 1

    print("the finalised parameters are {}, {}, with a security level of {}".format(n, q, security_level))

    return ZZ(n)


def automated_param_select_sd(n, sd=None, q=2 ** 32, reduction_cost_model=BKZ.sieve, secret_distribution=(0, 1),
                              target_security=128):
    """ A function used to generate the smallest value of sd which allows for 
    target_security bits of security, for the input values of (n,q)
    :param n: the LWE dimension
    :param q: the LWE modulus (q = 2**32, 2**64 in TFHE)
    :param reduction_cost_model: the BKZ cost model considered, BKZ.sieve is default
    :param secret_distribution: the LWE secret distribution
    :param target_security: the target number of bits of security, 128 is default

    EXAMPLE
    sage: X = automated_param_select_sd(n = 1054, q = 2**32)
    sage: X
    -26
    """

    if sd is None:
        # pick some random sd which gets us close (based on concrete_LWE_params)
        sd = round(n * 80 / (target_security * (-25)))

    sd_ = 2 ** sd * q
    alpha = sqrt(2 * pi) * sd_ / RR(q)

    # initial estimate, to determine if we are above or below the target security level
    print("estimating for n, q, sd = {}".format(log(sd_,2)))
    estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
    security_level = get_security_level(estimate)
    z = inequality(security_level, target_security)

    while z * security_level < z * target_security:
        sd += z * 1
        sd_ = 2 ** sd * q
        alpha = sqrt(2 * pi) * sd_ / RR(q)
        print("estimating for n, q, sd = {}".format(log(sd_,2)))
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
        security_level = get_security_level(estimate)

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        sd -= z * 1
        sd_ = 2 ** sd * q
        alpha = sqrt(2 * pi) * sd_ / RR(q)
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo)
        security_level = get_security_level(estimate)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(n,
                                                                                                                      sd,
                                                                                                                      log(q,
                                                                                                                          2),
                                                                                                                      security_level))

    return sd


def generate_parameter_matrix(n_range, sd=None, q=2 ** 32, reduction_cost_model=BKZ.sieve,
                              secret_distribution=(0, 1), target_security=128):
    """
    :param n_range: a tuple (n_min, n_max) giving the values of n for which to generate parameters
    :param sd: the standard deviation of the LWE error
    :param q: the LWE modulus (q = 2**32, 2**64 in TFHE)
    :param reduction_cost_model: the BKZ cost model considered, BKZ.sieve is default
    :param secret_distribution: the LWE secret distribution
    :param target_security: the target number of bits of security, 128 is default

    TODO: we should probably parallelise this function for speed
    TODO: code seems to fail when the initial estimate is < target_security bits

    EXAMPLE:
    sage: X = generate_parameter_matrix([788, 790])
    sage: X
    [(788, 4294967296, -20.0), (789, 4294967296, -20.0)]
    """

    RESULTS = []

    # grab min and max value/s of n
    (n_min, n_max) = n_range

    sd_ = sd

    for n in range(n_min, n_max):
        sd = automated_param_select_sd(n, sd=sd_, q=q, reduction_cost_model=reduction_cost_model,
                                       secret_distribution=secret_distribution, target_security=target_security)
        sd_ = sd
        RESULTS.append((n, q, sd))



    return RESULTS


def generate_parameter_step(results):
    """
    Plot results 
    :param results: an output of generate_parameter_matrix

    returns: a step plot of chosen parameters

    EXAMPLE:
    X = generate_parameter_matrix([700, 790])
    generate_parameter_step(X)
    plt.show()
    """

    N = []
    SD = []

    for (n, q, sd) in results:
        N.append(n)
        SD.append(sd)

    plt.scatter(N, SD)

    return plt


def test_rounded_gaussian(sigma, number_samples):
    """
    TODO: actually use a _rounded_ gaussian to match Concrete

    A function which simulates sampling from a Discrete Gaussian distribution
    :param sigma: the standard deviation
    :param number_samples: the number of samples to draw

    returns: a list of (value, count) pairs (essentially a histogram)

    EXAMPLE:

    sage: X = test_rounded_gaussian(2/3, 100000)
    sage: X
    [(-3, 2), (-2, 714), (-1, 19495), (0, 59658), (1, 19452), (2, 678), (3, 1)]
    """
 
    D = DiscreteGaussianDistributionIntegerSampler(sigma)
    samples = []
 
    for i in range(number_samples):
        samples.append(D())
 
    # now create a histogram
    hist = []
    for val in set(samples):
        hist.append((val, samples.count(val)))
 
    # sort (values)
    hist.sort(key=lambda x:x[0])
    return hist




