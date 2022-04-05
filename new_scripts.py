from estimator_new import *
from math import log2

def old_models(security_level, sd, logq = 32):
    """
    Use the old model as a starting point for the data gathering step
    TODO: update this and integrate a flag for it
    """

    def evaluate_model(sd, a, b):
        return (sd - b)/a

    models = dict()

    models["80"] = (-0.04049295502947623, 1.1288318226557081 + logq)
    models["96"] = (-0.03416314056943681, 1.4704806061716345 + logq)
    models["112"] = (-0.02970984362676178, 1.7848907787798667 + logq)
    models["128"] = (-0.026361288425133814, 2.0014671315214696 + logq)
    models["144"] = (-0.023744534465622812, 2.1710601038230712 + logq)
    models["160"] = (-0.021667220727651954, 2.3565507936475476 + logq)
    models["176"] = (-0.019947662046189942, 2.5109588704235803 + logq)
    models["192"] = (-0.018552804646747204, 2.7168913723130816 + logq)
    models["208"] = (-0.017291091126923574, 2.7956961446214326 + logq)
    models["224"] = (-0.016257546811508806, 2.9582401000615226 + logq)
    models["240"] = (-0.015329741032015766, 3.0744579055889782 + logq)
    models["256"] = (-0.014530554319171845, 3.2094375376751745 + logq)

    (a, b) = models["{}".format(security_level)]
    n_est = evaluate_model(sd, a, b)

    return round(n_est)


def estimate(params):
    """
    Retrieve an estimate using the Lattice Estimator, for a given set of input parameters
    :param params: the input LWE parameters
    """

    est = LWE.estimate(params, deny_list=("arora-gb", "bkw"))
    return est


def get_security_level(est, dp = 2):
    """
    Get the security level lambda from a Lattice Estimator output
    :param est: the Lattice Estimator output
    :param dp      : the number of decimal places to consider
    """
    attack_costs = []
    for key in est.keys():
        attack_costs.append(est[key]["rop"])
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
    :param target_security: the target number of bits of security, 128 is default

    EXAMPLE:
    sage: X = automated_param_select_n(Kyber512, target_security = 128)
    sage: X
    456
    """

    # get an initial estimate
    # costs = estimate(params)
    # security_level = get_security_level(costs, 2)
    # determine if we are above or below the target security level
    # z = inequality(security_level, target_security)

    # get an estimate based on the prev. model
    n_start = old_models(target_security, log2(params.Xe.stddev))
    params = params.updated(n=n_start)
    costs2 = estimate(params)
    security_level = get_security_level(costs2, 2)
    z = inequality(security_level, target_security)


    # we keep n > 2 * target_security as a rough baseline for mitm security (on binary key guessing)
    while z * security_level < z * target_security and params.n > 2 * target_security:
        params = params.updated(n = params.n + z * 8)
        costs = estimate(params)
        security_level = get_security_level(costs, 2)

        if -1 * params.Xe.stddev > 0:
            print("target security level is unatainable")
            break

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        params = params.updated(n = params.n - z * 8)
        costs = estimate(params)
        security_level = get_security_level(costs, 2)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(params.n,
                                                                                                                      log2(params.Xe.stddev),
                                                                                                                      log2(params.q),
                                                                                                                     security_level))

    # final sanity check so we don't return insecure (or inf) parameters
    # TODO: figure out inf in new estimator
    # or security_level == oo:
    if security_level < target_security:
        params.updated(n=None)

    return params

def generate_parameter_matrix(params_in, sd_range, target_security_levels=[128]):
    """
    :param sd_range: a tuple (sd_min, sd_max) giving the values of sd for which to generate parameters
    :param params: the standard deviation of the LWE error
    :param target_security: the target number of bits of security, 128 is default

    EXAMPLE:
    sage: X = generate_parameter_matrix()
    sage: X
    """

    results = dict()

    # grab min and max value/s of n
    (sd_min, sd_max) = sd_range

    n = params_in.n
    for lam in target_security_levels:
        results["{}".format(lam)] = []
        for sd in range(sd_min, sd_max + 1):
            Xe_new = nd.NoiseDistribution.DiscreteGaussian(2**sd)
            params_out = automated_param_select_n(params_in.updated(Xe=Xe_new), target_security=lam)
            results["{}".format(lam)].append((params_out.n, params_out.q, params_out.Xe.stddev))

    return results



def test_it():

    params = Kyber512

    # x = estimate(params)
    # y = get_security_level(x, 2)
    # print(y)
    #z1 = automated_param_select_n(schemes.TFHE630.updated(n=786), 128)
    #print(z1)
    sd_range = [1,4]
    z3 = generate_parameter_matrix(schemes.TFHE630, sd_range=[17,19], target_security_levels=[128, 192, 256])
    print(z3)

    return 0


def generate_zama_curves64():
    return 0


test_it()