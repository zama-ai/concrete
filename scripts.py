import estimator.estimator as est
import matplotlib.pyplot as plt
import numpy as np

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

# we add an enumeration model for completeness
cost_models = [classical, quantum, classical_conservative, quantum_conservative, est.BKZ.enum]


def estimate_lwe_nocrash(n, alpha, q, secret_distribution,
                                reduction_cost_model=est.BKZ.sieve, m=oo):
    """ 
    A function to estimate the complexity of LWE, whilst skipping over any attacks which crash
    :param n                   : the LWE dimension
    :param alpha               : the noise rate of the error
    :param q                   : the LWE ciphertext modulus
    :param secret_distribution : the LWE secret distribution
    :param reduction_cost_model: the BKZ reduction cost model
    :param m                   : the number of available LWE samples

    EXAMPLE:
    sage: estimate_lwe_nocrash(n = 256, q = 2**32, alpha = RR(8/2**32), secret_distribution = (0,1))
    sage: 39.46
    """

    # the success value denotes whether we need to re-run the estimator, in the case of a crash
    success = 0

    try:
        # we begin by trying all four attacks (usvp, dual, dec, mitm)
        estimate = est.estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo,
                                skip={"bkw", "dec", "arora-gb"})
        success = 1

    except Exception as e:
        print(e)
    
    if success == 0:
        try:
            # dual crashes most often, so try skipping dual first
            estimate = est.estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo,
                                    skip={"bkw", "dec", "arora-gb", "dual"})
            success = 1

        except Exception as e:
            print(e)

    if success == 0:
        try:
            # next, skip mitm
            estimate = est.estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                    reduction_cost_model=reduction_cost_model, m=oo,
                                    skip={"mitm", "bkw", "dec", "arora-gb", "dual"})
        
        except Exception as e:
            print(e)

    # the output security level is just the cost of the fastest attack
    security_level = get_security_level(estimate)

    return security_level



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

    try:
        levels.append(estimate["mitm"]["rop"])
    
    except:
        pass

    # take the minimum attack cost (in bits)
    security_level = round(log(min(levels), 2), decimal_places)

    return security_level


def inequality(x, y):
    """ A function which compresses the conditions
    x < y and x > y into a single condition via a 
    multiplier
    """
    if x <= y:
        return 1

    if x > y:
        return -1


def automated_param_select_n(sd, n=None, q=2 ** 32, reduction_cost_model=est.BKZ.sieve, secret_distribution=(0, 1),
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

    security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)

    z = inequality(security_level, target_security)

    while z * security_level < z * target_security and n > 80:
        n += z * 8
        alpha = sqrt(2 * pi) * sd / RR(q)
        security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)

        if (-1 * sd > 0):
            print("target security level is unatainable")
            break

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        n -= z * 8
        alpha = sqrt(2 * pi) * sd / RR(q)
        security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(n,
                                                                                                                      sd,
                                                                                                                      log(q,
                                                                                                                          2),
                                                                                                                     security_level))

    # final sanity check so we don't return insecure (or inf) parameters
    if security_level < target_security or security_level == oo:
        n = None

    return n


def automated_param_select_sd(n, sd=None, q=2**32, reduction_cost_model=est.BKZ.sieve, secret_distribution=(0, 1),
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
    security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)
    z = inequality(security_level, target_security)

    while z * security_level < z * target_security and sd > -log(q,2):

        sd += z * (0.5)
        sd_ = (2 ** sd) * q
        alpha = sqrt(2 * pi) * sd_ / RR(q)
        security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)

        ## THIS IS WHERE THE PROBLEM IS, CORRECT THIS CONDITION?
        if (sd > log(q, 2)):
            print("target security level is unatainable")
            return None

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        sd -= z * (0.5)
        sd_ = (2 ** sd) * q
        alpha = sqrt(2 * pi) * sd_ / RR(q)
        security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(n,
                                                                                                                      sd,
                                                                                                                      log(q,
                                                                                                                          2),
                                                                                                                      security_level))

    return sd


def generate_parameter_matrix(n_range, sd=None, q=2**32, reduction_cost_model=est.BKZ.sieve,
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

    # grab min and max value/s of n, with a granularity (if given as input)
    try:
        (n_min, n_max, gran) = n_range
    except:
        (n_min, n_max) = n_range
        gran = 1

    sd_ = sd

    for n in range(n_min, n_max + 1, gran):
        sd = automated_param_select_sd(n, sd=sd_, q=q, reduction_cost_model=reduction_cost_model,
                                       secret_distribution=secret_distribution, target_security=target_security)
        sd_ = sd
        RESULTS.append((n, q, sd))

    return RESULTS


def generate_parameter_matrix_sd(sd_range, n=None, q=2**32, reduction_cost_model=est.BKZ.sieve,
                              secret_distribution=(0, 1), target_security=128):
    """
    :param sd_range: a tuple (sd_min, sd_max) giving the values of sd for which to generate parameters
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


# dual bug example
# n = 256; q = 2**32; sd = 2**(-4); reduction_cost_model = BKZ.sieve
# _ = estimate_lwe(n, alpha, q, reduction_cost_model)

def test_params(n, q, sd, secret_distribution):

    sd = sd * q
    alpha = RR(sqrt(2*pi) * sd / q)

    est = est.estimate_lwe(n, alpha, q, secret_distribution = secret_distribution, reduction_cost_model = est.BKZ.sieve, skip = ("arora-gb", "bkw", "dec"))

    return est


def generate_iso_lines(N = [256, 2048], SD = [0, 32], q = 2**32):

    RESULTS = []

    for n in range(N[0], N[1] + 1, 1):
        for sd in range(SD[0], SD[1] + 1, 1):
            sd = 2**sd
            alpha = sqrt(2*pi) * sd / q
            try:
                est = est.estimate_lwe(n, alpha, q, secret_distribution = (0,1), reduction_cost_model = est.BKZ.sieve, skip = ("bkw", "arora-gb", "dec"))
                est = get_security_level(est, 2)
            except:
                est = est.estimate_lwe(n, alpha, q, secret_distribution = (0,1), reduction_cost_model = est.BKZ.sieve, skip = ("bkw", "arora-gb", "dual", "dec"))
                est = get_security_level(est, 2)
            RESULTS.append((n, sd, est))

    return RESULTS


def plot_iso_lines(results):

    x1 = []
    x2 = []

    for z in results:
        x1.append(z[0])
        # use log(q)
        # use -ve values to match Pascal's diagram
        x2.append(z[2])

    plt.plot(x1, x2)

    return plt


def test_multiple_sd(n, q, secret_distribution, reduction_cost_model, split = 33, m = oo):
     est = []
     Y = []
     for sd_ in np.linspace(0,32,split):
         Y.append(sd_)
         sd = (2** (-1 * sd_))* q
         alpha = sqrt(2*pi) * sd / q
         try:
             es = est.estimate_lwe(n=512, alpha=alpha, q=q, secret_distribution=(0, 1), reduction_cost_model = reduction_cost_model,
                                            skip=("bkw", "dec", "arora-gb"), m = m)
         except:
             es = est.estimate_lwe(n=512, alpha=alpha, q=q, secret_distribution=(0, 1), reduction_cost_model = reduction_cost_model,
                                            skip=("bkw", "dec", "arora-gb", "dual"), m = m)
         est.append(get_security_level(es,2))

     return est, Y


## parameter curves

def get_parameter_curves_data_sd(sec_levels, sd_range, q):

    Results = []
    for sec in sec_levels:
        try:
            result_sec = generate_parameter_matrix_sd(n = None, sd_range=sd_range, q=q, reduction_cost_model=est.BKZ.sieve,
                              secret_distribution=(0,1), target_security=sec)
            Results.append(result_sec)
        except:
            pass

    return Results


def get_parameter_curves_data_n(sec_levels, n_range, q):

    Results = []
    for sec in sec_levels:
        try:
            result_sec = generate_parameter_matrix(n_range,  sd = None, q=q, reduction_cost_model=est.BKZ.sieve,
                              secret_distribution=(0,1), target_security=sec)
            Results.append(result_sec)
        except:
            pass

    return Results


def interpolate_result(result, log_q):

    # linear function interpolation
    x = []
    y = []

    # 1. filter out any points which reccomend sd = -log(q) + 2
    new_result= []
    for res in result:
        if res[2] >= - log_q + 2:
            new_result.append(res)

    result = new_result
    for res in result:
        x.append(res[0])
        y.append(res[2])


    (a,b) = np.polyfit(x, y, 1)

    return (a,b)


def plot_interpolants(interpolants, n_range, log_q, degree = 1):
    for x in interpolants:
        if degree == 1:
            vals = [x[0] * n + x[1] for n in range(n_range[0],n_range[1])]
        elif degree == 2:
            vals = [x[0] * n**2 + x[1]*n + x[2] for n in range(n_range[0],n_range[1])]
        # any values which fall outside of the range and edited to give at least two bits of noise.

        vvals = []
        for v in vals:
            if v < -log_q + 2:
                vvals.append(-log_q + 2)
            else:
                vvals.append(v) 

        plt.plot(range(n_range[0], n_range[0] + len(vvals)), vvals)

    return 0


## currently running
# sage: n_range = (256, 2048, 16)
# sage: sec_levels = [80 + 16*k for k in range(0,12)]
# sage: results = get_parameter_curves_data_n(sec_levels, n_range, q = 2**64)

def verify_results(results, security_level, secret_distribution = (0,1), reduction_cost_model = est.BKZ.sieve):
    """ A function which verifies that a set of results match a given security level
    :param results       : a set of tuples of the form (n, q, sd)
    :param security_level: the target security level for these params
    """

    estimates = []

    # 1. Grab the parameters
    for (n, q, sd) in results:
        if sd is not None:
            sd = 2**sd
            alpha = sqrt(2*pi) * sd

            # 2. Test that these parameters satisfy the given security level
            try:
                estimate = est.estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                        reduction_cost_model=reduction_cost_model, m=oo, skip = {"bkw","dec","arora-gb"})
            except:
                estimate = est.estimate_lwe(n, alpha, q, secret_distribution=secret_distribution,
                                        reduction_cost_model=reduction_cost_model, m=oo,
                                        skip={"bkw", "dec", "arora-gb", "dual"})
        
            estimates.append(estimate)

    return estimates


def verify_interpolants(interpolant, n_range, log_q, secret_distribution = (0,1), reduction_cost_model = est.BKZ.sieve):

    estimates = []
    q = 2**log_q
    (a, b) = interpolant

    for n in range(n_range[0], n_range[1]):
        print(n)
        # we take the max here to ensure that the "cut-off" for the minimal error is met.
        sd = max(a * n + b, -log_q + 2)
        sd = 2 ** sd
        alpha = sqrt(2*pi) * sd

        security_level = estimate_lwe_nocrash(n, alpha, q, secret_distribution=secret_distribution,
                                reduction_cost_model=reduction_cost_model, m=oo)
        print(security_level)
        if security_level == oo:
            security_level = 0
        estimates.append(security_level)

    return estimates

def get_zama_curves():

    # hardcode the parameters for now
    n_range = [128, 2048, 32]
    sec_levels = [80 + 16*k for k in range(0,12)]
    results = get_parameter_curves_data_n(sec_levels, n_range, q = 2**64)

    return results


def test_curves():

    # a small hardcoded example for testing purposes

    n_range = [256, 1024, 128]
    sec_levels = [80, 128, 256]
    results = get_parameter_curves_data_n(sec_levels, n_range, q = 2**64)

    return results

def find_nalpha(l, sec_lvl):
    for j in range(len(l)):
        if l[j] != oo and l[j] > sec_lvl:
            return j


## we start with 80/128/192/256-bits of security

## lambda = 80
## z = verify_interpolants(interps[0], (128,2048), 64)
## i = 0
## min(z[i:]) = 80.36
## so the model is sd(n) = max(-0.04047677865612648 * n + 1.1433465085639063, log_q - 2), n >= 128


## lambda = 128
## z = verify_interpolants(interps[3], (128,2048), 64)
## i = 83
## min(z[i:]) = 128.02
## so the model is sd(n) = max(-0.026374888765705498 * n + 2.012143923330495, log_q - 2), n >= 211 ( = 128 + 83)


## lambda = 192
## z = verify_interpolants(interps[7], (128,2048), 64)
## i = 304
## min(z[i:]) = 192.19
## so the model is sd(n) = max(-0.018504919354426233 * n +  2.6634073426215843, log_q - 2), n >= 432 ( = 128 + 212)


## lambda = 256
## z = verify_interpolants(interps[-1], (128,2048), 64)
## i = 653
## min(z[i:]) = 256.25
## so the model is sd(n) = max(-0.014327640360322604 * n + 2.899270827311091), log_q - 2), n >= 781 ( = 128 + 653)
