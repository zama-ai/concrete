import sys
from estimator_new import *
from sage.all import oo, save, load, ceil
from math import log2
import multiprocessing


def old_models(security_level, sd, logq=32):
    """
    Use the old model as a starting point for the data gathering step
    :param security_level: the security level under consideration
    :param sd            : the standard deviation of the LWE error distribution Xe
    :param logq          : the (base 2 log) value of the LWE modulus q
    """

    def evaluate_model(a, b, stddev=sd):
        return (stddev - b)/a

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
    n_est = evaluate_model(a, b, sd)

    return round(n_est)


def estimate(params, red_cost_model=RC.BDGL16, skip=("arora-gb", "bkw")):
    """
    Retrieve an estimate using the Lattice Estimator, for a given set of input parameters
    :param params: the input LWE parameters
    :param red_cost_model: the lattice reduction cost model
    :param skip: attacks to skip
    """

    est = LWE.estimate(params, red_cost_model=red_cost_model, deny_list=skip)

    return est


def get_security_level(est, dp=2):
    """
    Get the security level lambda from a Lattice Estimator output
    :param est: the Lattice Estimator output
    :param dp: the number of decimal places to consider
    """
    attack_costs = []
    # note: key does not need to be specified est vs est.keys()
    for key in est:
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

    # get an estimate based on the prev. model
    print("n = {}".format(params.n))
    n_start = old_models(target_security, log2(
        params.Xe.stddev), log2(params.q))
    # n_start = max(n_start, 450)
    # TODO: think about throwing an error if the required n < 450

    params = params.updated(n=n_start)
    costs2 = estimate(params)
    security_level = get_security_level(costs2, 2)
    z = inequality(security_level, target_security)

    # we keep n > 2 * target_security as a rough baseline for mitm security (on binary key guessing)
    while z * security_level < z * target_security:
        # TODO: fill in this case! For n > 1024 we only need to consider every 256 (optimization)
        params = params.updated(n=params.n + z * 8)
        costs = estimate(params)
        security_level = get_security_level(costs, 2)

        if -1 * params.Xe.stddev > 0:
            print("target security level is unattainable")
            break

    # final estimate (we went too far in the above loop)
    if security_level < target_security:
        # we make n larger
        print("we make n larger")
        params = params.updated(n=params.n + 8)
        costs = estimate(params)
        security_level = get_security_level(costs, 2)

    print("the finalised parameters are n = {}, log2(sd) = {}, log2(q) = {}, with a security level of {}-bits".format(params.n,
                                                                                                                      log2(
                                                                                                                          params.Xe.stddev),
                                                                                                                      log2(
                                                                                                                          params.q),
                                                                                                                      security_level))

    if security_level < target_security:
        params.updated(n=None)

    return params, security_level


def generate_parameter_matrix(params_in, sd_range, target_security_levels=[128], name="default_name"):
    """
    :param params_in: a initial set of LWE parameters
    :param sd_range: a tuple (sd_min, sd_max) giving the values of sd for which to generate parameters
    :param target_security_levels: a list of the target number of bits of security, 128 is default
    :param name: a name to save the file
    """

    (sd_min, sd_max) = sd_range
    for lam in target_security_levels:
        for sd in range(sd_min, sd_max + 1):
            print("run for {}".format(lam, sd))
            Xe_new = nd.NoiseDistribution.DiscreteGaussian(2**sd)
            (params_out, sec) = automated_param_select_n(
                params_in.updated(Xe=Xe_new), target_security=lam)

            try:
                results = load("{}.sobj".format(name))
            except:
                results = dict()
                results["{}".format(lam)] = []

            results["{}".format(lam)].append(
                (params_out.n, log2(params_out.q), log2(params_out.Xe.stddev), sec))
            save(results, "{}.sobj".format(name))

    return results


def generate_zama_curves64(sd_range=[2, 58], target_security_levels=[128], name="default_name"):
    """
    The top level function which we use to run the experiment

    :param sd_range: a tuple (sd_min, sd_max) giving the values of sd for which to generate parameters
    :param target_security_levels: a list of the target number of bits of security, 128 is default
    :param name: a name to save the file
    """
    if __name__ == '__main__':

        D = ND.DiscreteGaussian
        vals = range(sd_range[0], sd_range[1])
        pool = multiprocessing.Pool(2)
        init_params = LWE.Parameters(
            n=1024, q=2 ** 64, Xs=D(0.50, -0.50), Xe=D(2 ** 55), m=oo, tag='params')
        inputs = [(init_params, (val, val), target_security_levels, name)
                  for val in vals]
        res = pool.starmap(generate_parameter_matrix, inputs)

    return "done"


# The script runs the following commands
# grab values of the command-line input arguments
a = int(sys.argv[1])
b = int(sys.argv[2])
c = int(sys.argv[3])
# run the code
generate_zama_curves64(sd_range=(b, c), target_security_levels=[
                       a], name="{}".format(a))
