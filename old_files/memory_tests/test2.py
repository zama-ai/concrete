
from multiprocessing import *
from estimator_new import *
from sage.all import oo, save


def test_memory(x):
    print("doing job...")
    print(x)
    y = LWE.estimate(x, deny_list=("arora-gb", "bkw"))
    return y

if __name__ == "__main__":
    D = ND.DiscreteGaussian
    params = LWE.Parameters(n=1024, q=2 ** 64, Xs=D(0.50, -0.50), Xe=D(2**57), m=oo, tag='TFHE_DEFAULT')

    names = [params, params.updated(n=761), params.updated(q=2**65), params.updated(n=762)]
    procs = []
    proc = Process(target=print_func)
    procs.append(proc)
    proc.start()
    p = Pool(1)

    for name in names:
        proc = Process(target=test_memory, args=(name,))
        procs.append(proc)
        proc.start()
        proc.join()

    for proc in procs:
        proc.join()