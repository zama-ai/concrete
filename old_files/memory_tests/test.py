from estimator_new import *
from sage.all import oo, save

def test():

    # code
    D = ND.DiscreteGaussian
    params = LWE.Parameters(n=1024, q=2 ** 64, Xs=D(0.50, -0.50), Xe=D(2**57), m=oo, tag='TFHE_DEFAULT')

    names = [params, params.updated(n=761), params.updated(q=2 ** 65), params.updated(n=762)]

    for name in names:
        x = LWE.estimate(name, deny_list=("arora-gb", "bkw"))

    return 0

test()