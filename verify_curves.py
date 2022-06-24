import numpy as np
from sage.all import save, load, ceil


def sort_data(security_level):
    from operator import itemgetter

    # step 1. load the data
    X = load("{}.sobj".format(security_level))

    # step 2. sort by SD
    x = sorted(X["{}".format(security_level)], key=itemgetter(2))

    # step3. replace the sorted value
    X["{}".format(security_level)] = x

    return X


def generate_curve(security_level):

    # step 1. get the data
    X = sort_data(security_level)

    # step 2. group the n and sigma data into lists
    N = []
    SD = []
    for x in X["{}".format(security_level)]:
        N.append(x[0])
        SD.append(x[2] + 0.5)

    # step 3. perform interpolation and return coefficients
    (a, b) = np.polyfit(N, SD, 1)

    return a, b


def verify_curve(security_level, a=None, b=None):

    # step 1. get the table and max values of n, sd
    X = sort_data(security_level)
    n_max = X["{}".format(security_level)][0][0]

    # step 2. a function to get model values
    def f_model(a, b, n):
        return ceil(a * n + b)

    # step 3. a function to get table values
    def f_table(table, n):
        for i in range(len(table)):
            n_val = table[i][0]
            if n < n_val:
                pass
            else:
                j = i
                break

        # now j is the correct index, we return the corresponding sd
        return table[j][2]

    # step 3. for each n, check whether we satisfy the table
    n_min = max(2 * security_level, 450, X["{}".format(security_level)][-1][0])
    print(n_min)
    print(n_max)

    for n in range(n_max, n_min, - 1):
        model_sd = f_model(a, b, n)
        table_sd = f_table(X["{}".format(security_level)], n)
        print(n, table_sd, model_sd, model_sd >= table_sd)

        if table_sd > model_sd:
            print("MODEL FAILS at n = {}".format(n))
            return "FAIL"

    return "PASS", n_min


def generate_and_verify(security_levels, log_q, name="verified_curves"):

    data = []

    for sec in security_levels:
        print("WE GO FOR {}".format(sec))
        # generate the model for security level sec
        (a_sec, b_sec) = generate_curve(sec)
        # verify the model for security level sec
        res = verify_curve(sec, a_sec, b_sec)
        # append the information into a list
        data.append((a_sec, b_sec - log_q, sec, res[0], res[1]))
        save(data, "{}.sobj".format(name))

    return data


data = generate_and_verify(
    [80, 96, 112, 128, 144, 160, 176, 192, 256], log_q=64)
print(data)
