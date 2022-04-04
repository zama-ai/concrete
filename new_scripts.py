## for now, run using sage new_scripts.py
from estimator_new import *
from math import log2

def estimate_lwe_nocrash(params):
    """
    Retrieve an estimate using the Lattice Estimator, for a given set of input parameters
    :param params: the input LWE parameters
    """
    try:
        estimate = LWE.estimate(params)
    except:
        estimate = 0
    return estimate

def get_security_level(estimate, dp = 2):
    """
    Get the security level lambda from a Lattice Estimator output
    :param estimate: the Lattice Estimator output
    :param dp      : the number of decimal places to consider
    """
    attack_costs = []
    for key in estimate.keys():
        attack_costs.append(estimate[key]["rop"])
    # get the security level correct to 'dp' decimal places
    security_level = round(log2(min(attack_costs)), dp)
    return security_level

def inequality(x, y):
    """ A utility function which compresses the conditions x < y and x > y into a single condition via a multiplier
    :param x: the LHS of the inequality
    :param y: the RHS of the inequality
    """
    if x <= y:
        return 1

    if x > y:
        return -1

def automated_param_select_n(params, target_security=128):
    """ A function used to generate the smallest value of n which allows for
    target_security bits of security, for the input values of (params.Xe.stddev,params.q)
    :param params: the standard deviation of the error
    :param secret_distribution: the LWE secret distribution
    :param target_security: the target number of bits of security, 128 is default

    EXAMPLE:
    sage: X = automated_param_select_n(Kyber512, target_security = 128)
    sage: X
    456
    """

    # get an initial estimate
    costs = estimate_lwe_nocrash(params)
    security_level = get_security_level(costs, 2)
    # determine if we are above or below the target security level
    z = inequality(security_level, target_security)

    while z * security_level < z * target_security and params.n > 80:
        params = params.updated(n = params.n + z * 8)
        costs = estimate_lwe_nocrash(params)
        security_level = get_security_level(costs, 2)

        if (-1 * params.Xe.stddev > 0):
            print("target security level is unatainable")
            break

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        params = params.updated(n = params.n - z * 8)
        costs = estimate_lwe_nocrash(params)
        security_level = get_security_level(costs, 2)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(params.n,
                                                                                                                      params.Xe.stddev,
                                                                                                                      log2(params.q),
                                                                                                                     security_level))

    # final sanity check so we don't return insecure (or inf) parameters
    # TODO: figure out inf in new estimator
    if security_level < target_security: #or security_level == oo:
        params.update(n = None)

    return params

def automated_param_select_sd(params, target_security=128):
    """ A function used to generate the smallest value of sd which allows for
    target_security bits of security, for the input values of (params.Xe.stddev,params.q)
    :param params: the standard deviation of the error
    :param secret_distribution: the LWE secret distribution
    :param target_security: the target number of bits of security, 128 is default

    EXAMPLE:
    sage: X = automated_param_select_n(Kyber512, target_security = 128)
    sage: X
    456
    """

    #if sd is None:
        # pick some random sd which gets us close (based on concrete_LWE_params)
    #    sd = round(n * 80 / (target_security * (-25)))

        # make sure sd satisfies q * sd > 1
    #    sd = max(sd, -(log(q,2) - 2))

    #sd_ = (2 ** sd) * q
    #alpha = sqrt(2 * pi) * sd_ / RR(q)

    # get an initial estimate
    costs = estimate_lwe_nocrash(params)
    security_level = get_security_level(costs, 2)
    # determine if we are above or below the target security level
    z = inequality(security_level, target_security)

    while z * security_level < z * target_security: #and sd > -log(q,2):

        Xe_new = nd.NoiseDistribution.DiscreteGaussian(params.Xe.stddev + z * (0.5))
        params.updated(Xe = Xe_new)
        costs = estimate_lwe_nocrash(params)
        security_level = get_security_level(costs, 2)

        if (params.Xe.stddev > log2(params.q)):
            print("target security level is unatainable")
            return None

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        Xe_new = nd.NoiseDistribution.DiscreteGaussian(params.Xe.stddev - z * (0.5))
        params.updated(Xe = Xe_new)
        costs = estimate_lwe_nocrash(params)
        security_level = get_security_level(costs, 2)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(params.n,
                                                                                                                      params.Xe.stddev,
                                                                                                                      log2(params.q),
                                                                                                                      security_level))

    return params



params = Kyber512

#x = estimate_lwe_nocrash(params)
#y = get_security_level(x, 2)
#print(y)
#z1 = automated_param_select_n(Kyber512, 128)
#print(z1)
#z2 = automated_param_select_sd(Kyber512, 128)
#print(z2)