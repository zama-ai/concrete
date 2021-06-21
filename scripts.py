import matplotlib.pyplot as plt
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionIntegerSampler
from concrete_params import concrete_LWE_params, concrete_RLWE_params
import numpy as np
from pytablewriter import MarkdownTableWriter
from hybrid_decoding import parameter_search
from random import uniform
from mpl_toolkits import mplot3d
# easier to just load the estimator
load("estimator.py")

# define the four cost models used for Concrete (2 classical, 2 quantum)
# note that classical and quantum are the two models used in the "HE Std"


def classical(beta, d, B):
    return ZZ(2) ** RR(0.292 * beta + 16.4 + log(8 * d, 2))


def quantum(beta, d, B):
    return ZZ(2) ** RR(0.265 * beta + 16.4 + log(8 * d, 2))


def classical_conservative(beta, d, B):
    return ZZ(2) ** RR(0.292 * beta)


def quantum_conservative(beta, d, B):
    return ZZ(2) ** RR(0.265 * beta)


cost_models = [classical, quantum, classical_conservative, quantum_conservative, BKZ.enum]

# functions to automate parameter selection

def get_security_level(estimate, decimal_places = 2):
    """ Function to get the security level from an LWE Estimator output, 
    i.e. returns only the bit-security level (without the attack params)
    :param estimate: the input estimate
    :param decimal_places: the number of decimal places

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
        # assume access to an infinite number of samples
        m = oo

        for model in cost_models:
            try:
                model = model[0]
            except:
                model = model
            estimate = parameter_search(mitm = True, reduction_cost_model = est.BKZ.sieve, n = n, q = q, alpha = alpha, m = m, secret_distribution = secret_distribution)
            results.append(get_security_level(estimate))

        RESULTS.append(results)

    return RESULTS

def get_hybrid_security_levels(params):
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

    return latex(table(results))


def markdownit(results, headings = ["Parameter Set", "Classical", "Quantum", "Classical (c)", "Quantum (c)", "Enum"]):
    """
    A function which takes the output of get_all_security_levels() and
    turns it into a markdown table
    :param results: the security levels

    sage: X = get_all_security_levels(concrete_LWE_params)
    sage: markdownit(X)
    # estimates
    |Parameter Set|Classical|Quantum|Classical (c)|Quantum (c)| Enum |
    |-------------|---------|-------|-------------|-----------|------|
    |LWE128_256   |126.69   |117.57 |98.7         |89.57      |217.55|
    |LWE128_512   |135.77   |125.92 |106.58       |96.73      |218.53|
    |LWE128_638   |135.27   |125.49 |105.7        |95.93      |216.81|
    [...]
    """

    writer = MarkdownTableWriter(value_matrix = results, headers = headings, table_name = "estimates")
    writer.write_table()

    return writer


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
    :param n: an initial value of n to use in optimisation, guessed if None
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
    print("estimating for n = {}, q, sd".format(n))
    try:
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo,
                                skip={"bkw", "dec", "arora-gb", "mitm"})
    except Exception as e:
        print(e)
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo,
                                skip={"bkw", "dec", "arora-gb", "mitm", "dual"})
        print("the estimate is {}".format(estimate))
    security_level = get_security_level(estimate)
    print("the security level is: {}".format(security_level))
    z = inequality(security_level, target_security)
    print("the result of Z is{}".format(z))

    while z * security_level < z * target_security and n > 80:
        n += z * 8
        print("N = {}".format(n))
        print("SECURITY LEVEL = {}".format(security_level))
        alpha = sqrt(2 * pi) * sd / RR(q)
        print("estimating for n = {}, q, sd".format(n))
        try:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
        except:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo,
                                    skip={"bkw", "dec", "arora-gb", "mitm", "dual"})
        security_level = get_security_level(estimate)

        if (-1 * sd > log(q, 2)):
            print("target security level is unatainable")
            break

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        n -= z * 8
        alpha = sqrt(2 * pi) * sd / RR(q)
        print("N = {}".format(n))
        print("SECURITY LEVEL = {}".format(security_level))
        try:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
        except:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo,
                                    skip={"bkw", "dec", "arora-gb", "mitm", "dual"})
        security_level = get_security_level(estimate)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(n,
                                                                                                                      sd,
                                                                                                                      log(q,
                                                                                                                          2),
                                                                                                                     security_level))

    # final sanity check so we don't return insecure (or inf) parameters
    if security_level < target_security or security_level == oo:
        n = None

    return n



def automated_param_select_sd(n, sd=None, q=2**32, reduction_cost_model=BKZ.sieve, secret_distribution=(0, 1),
                              target_security=128):
    """ A function used to generate the smallest value of sd which allows for 
    target_security bits of security, for the input values of (n,q)
    :param n: the LWE dimension
    :param sd: an initial value of sd to use in optimisation, guessed if None
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

        # make sure sd satisfies q * sd > 1
        sd = max(sd, -(log(q,2) - 2))

    sd_ = (2 ** sd) * q
    alpha = sqrt(2 * pi) * sd_ / RR(q)

    # initial estimate, to determine if we are above or below the target security level
    print("estimating for n, q, sd = {}".format(log(sd_,2)))
    try:
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo,
                                skip={"bkw", "dec", "arora-gb", "mitm"})
    except:
        estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo,
                                skip={"bkw", "dec", "arora-gb", "mitm", "dual"})
    security_level = get_security_level(estimate)
    z = inequality(security_level, target_security)

    while z * security_level < z * target_security and sd > -log(q,2):
        sd += z * 1
        sd_ = (2 ** sd) * q
        alpha = sqrt(2 * pi) * sd_ / RR(q)
        print("estimating for n, q, sd = {}".format(log(sd_,2)))
        try:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
        except:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo,
                                    skip={"bkw", "dec", "arora-gb", "mitm", "dual"})
        security_level = get_security_level(estimate)

        if (-1 * sd > log(q, 2)):
            print("target security level is unatainable")
            break

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        sd -= z * 1
        sd_ = (2 ** sd) * q
        alpha = sqrt(2 * pi) * sd_ / RR(q)
        try:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb","mitm"})
        except:
            estimate = estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo,
                                    skip={"bkw", "dec", "arora-gb", "mitm", "dual"})
        security_level = get_security_level(estimate)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(n,
                                                                                                                      sd,
                                                                                                                      log(q,
                                                                                                                          2),
                                                                                                                      security_level))

    return sd


def generate_parameter_matrix(n_range, sd=None, q=2**32, reduction_cost_model=BKZ.sieve,
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

    for n in range(n_min, n_max + 1):
        sd = automated_param_select_sd(n, sd=sd_, q=q, reduction_cost_model=reduction_cost_model,
                                       secret_distribution=secret_distribution, target_security=target_security)
        sd_ = sd
        RESULTS.append((n, q, sd))

    return RESULTS


def generate_parameter_matrix_sd(sd_range, n=None, q=2**32, reduction_cost_model=BKZ.sieve,
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
    (sd_min, sd_max) = sd_range

    n = n

    for sd in range(sd_min, sd_max + 1):
        n = automated_param_select_n(sd, n=n, q=q, reduction_cost_model=reduction_cost_model,
                                       secret_distribution=secret_distribution, target_security=target_security)
        RESULTS.append((n, q, sd))

    return RESULTS


def generate_parameter_step(results, label = None, torus_sd = True):
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
        if torus_sd:
            SD.append(sd)
        else:
            SD.append(sd + log(q,2))

    plt.plot(N, SD, label = label)
    plt.legend(loc = "upper right")

    return plt


def test_rounded_gaussian(sigma, number_samples, q = None):
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
        if q:
            samples.append(D() % q)
        else:
            samples.append(D())
    # now create a histogram
    hist = []
    for val in set(samples):
        hist.append((val, samples.count(val)))
 
    # sort (values)
    hist.sort(key=lambda x:x[0])
    return hist


def test_uniform(number_samples, q):
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

    samples = []

    for i in range(number_samples):
        samples.append(round(uniform(0, q)))
    # now create a histogram
    hist = []
    for val in set(samples):
        hist.append((val, samples.count(val)))

    # sort (values)
    hist.sort(key=lambda x: x[0])
    return hist

# dual bug example
# n = 256; q = 2**32; sd = 2**(-4); reduction_cost_model = BKZ.sieve
# _ = estimate_lwe(n, alpha, q, reduction_cost_model)

def test_params(n, q, sd, secret_distribution):

    sd = sd * q
    alpha = RR(sqrt(2*pi) * sd / q)

    est = estimate_lwe(n, alpha, q, secret_distribution = secret_distribution, reduction_cost_model = BKZ.sieve, skip = ("arora-gb", "bkw", "mitm", "dec"))

    return est


def generate_iso_lines(N = [256, 2048], SD = [0, 32], q = 2**32):

    RESULTS = []

    for n in range(N[0], N[1] + 1, 1):
        for sd in range(SD[0], SD[1] + 1, 1):
            sd = 2**sd
            alpha = sqrt(2*pi) * sd / q
            try:
                est = estimate_lwe(n, alpha, q, secret_distribution = (0,1), reduction_cost_model = BKZ.sieve, skip = ("bkw", "mitm", "arora-gb", "dec"))
                est = get_security_level(est, 2)
            except:
                est = estimate_lwe(n, alpha, q, secret_distribution = (0,1), reduction_cost_model = BKZ.sieve, skip = ("bkw", "mitm", "arora-gb", "dual", "dec"))
                est = get_security_level(est, 2)
            RESULTS.append((n, sd, est))

    return RESULTS


def plot_iso_lines(results):

    x1 = []
    x2 = []
    x3 = []

    for z in results:
        x1.append(z[0])
        # use log(q)
        # use -ve values to match Pascal's diagram
        x2.append(-1 * log(z[1],2))
        x3.append(z[3])

    plt.scatter(x1, x2, c = x3)
    plt.colorbar()

    return plt


def test_multiple_sd(n, q, secret_distribution, reduction_cost_model, split = 33, m = oo):
     est = []
     Y = []
     for sd_ in np.linspace(0,32,split):
         Y.append(sd_)
         sd = (2** (-1 * sd_))* q
         alpha = sqrt(2*pi) * sd / q
         try:
             es = estimate_lwe(n=512, alpha=alpha, q=q, secret_distribution=(0, 1), reduction_cost_model = reduction_cost_model,
                                            skip=("bkw", "mitm", "dec", "arora-gb"), m = m)
         except:
             print("except")
             es = estimate_lwe(n=512, alpha=alpha, q=q, secret_distribution=(0, 1), reduction_cost_model = reduction_cost_model,
                                            skip=("bkw", "mitm", "dec", "arora-gb", "dual"), m = m)
         est.append(get_security_level(es,2))

     return est, Y


def output_secret_distribution(m):
    """
    generate the correct secret_distirbution for the given input
    :param m: the number of elements in the secret distribution
    """

    # the code doesn't work for m < 2
    assert m >= 2

    if m % 2 ==1:
        # m is odd
        b = (m - 1)/2
        secret_distribution = (-b, b)
    else:
        # m is even
        b = m / 2 - 1
        secret_distribution = (-b, b + 1)

    return secret_distribution

def get_marcs_curves(n_range, q, m_max):

    # the final result will be a list of m_max elements, each containing
    # a parameter matrix
    RESULTS = []

    for m in range(2, m_max + 1):
        secret_distribution = output_secret_distribution(m)
        result_m = generate_parameter_matrix(n_range, sd=None, q=q, reduction_cost_model=BKZ.sieve,
                              secret_distribution=secret_distribution, target_security=128)
        RESULTS.append(result_m)
    return RESULTS


def get_marcs_curves_n(sd_range, q, m_max):

    # the final result will be a list of m_max elements, each containing
    # a parameter matrix
    RESULTS = []

    for m in range(2, m_max + 1):
        secret_distribution = output_secret_distribution(m)
        result_m = generate_parameter_matrix_sd(n = None, sd_range=sd_range, q=q, reduction_cost_model=BKZ.sieve,
                              secret_distribution=secret_distribution, target_security=128)
        RESULTS.append(result_m)
    return RESULTS


def tabulate_results(results):
    """ Put the results from get_marcs_curves into a LaTeX table
    """

    new_results = []
    num_results = len(results[0])
    num_entries = len(results)

    key = []
    key.append("n")
    key.append("q")
    for i in range(num_entries):
        key.append("m = {}".format(i + 2))
    
    new_results.append(key)




    for j in range(num_results):
        result_j = []
        result_j.append(results[0][j][0])
        result_j.append(int(log(results[0][0][1],2)))

        for i in range(num_entries):
            result_j.append(int(results[i][j][2]))
        
        new_results.append(result_j)

    return new_results

def tabulate_results_sd(results):

    new_results = []
    num_results = len(results[0])
    num_entries = len(results)

    key = []
    key.append("sd")
    key.append("q")

    for i in range(num_entries):
        key.append("m = {}".format(i + 2))

    new_results.append(key)

    for j in range(num_results):
        result_j = []
        result_j.append(results[0][j][2])
        result_j.append(int(log(results[0][0][1],2)))

        for i in range(num_entries):
            try:
                result_j.append(int(results[i][j][0]))
            except:
                result_j.append(str(results[i][j][0]))

        new_results.append(result_j)

    return new_results

# code to cross-check the security levels for marc/pascal results
# sage: with open("results_32_128.txt", "rb") as fp:   # Unpickling
# ....: ...   X = pickle.load(fp)
# res = []
# sage: for i in range(len(X)):
# ....:     x = X[i]
# ....:     m = i + 2
# ....:     secret_distribution = output_secret_distribution(m)
# ....:     for (n, q, sd) in x:
# ....:         if n is not None:
# ....:             sd = 2**(sd)
# ....:             alpha = sqrt(2*pi) * sd
# ....:             print((n, q, sd))
# ....:             try:
# ....:                 _ = estimate_lwe(n, alpha, q, secret_distribution = secret_distribution, reduction_cost_model = BKZ.sieve, skip = ("arora-gb", "mitm", "bkw", "dec"))
# ....:             except:
# ....:                 _ = estimate_lwe(n, alpha, q, secret_distribution = secret_distribution, reduction_cost_model = BKZ.sieve, skip = ("arora-gb", "mitm", "bkw", "dec", "dual"))
# ....:         else:
# ....:             print("None")
# ....:         res.append(get_security_level(_))
# ....:print(min(res))