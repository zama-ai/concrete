import estimator.estimator as est
from concrete_params import concrete_LWE_params, concrete_RLWE_params
from hybrid_decoding import parameter_search

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

# dual bug example
# n = 256; q = 2**32; sd = 2**(-4); reduction_cost_model = BKZ.sieve
# _ = estimate_lwe(n, alpha, q, reduction_cost_model)