# Security

This document describes some security concepts around FHE that can help you generate parameters that are both secure and correct.

## Parameter Curves

To select secure cryptographic parameters for usage in Concrete, we utilize the [Lattice-Estimator](https://github.com/malb/lattice-estimator). In particular, we use the following workflow:

1. Data Acquisition
    - For a given value of $$(n, q = 2^{64}, \sigma)$$ we obtain raw data from the Lattice Estimator, which ultimately leads to a security level $$\lambda$$. All relevant attacks in the Lattice Estimator are considered.
    - Increase the value of $$\sigma$$, until the tuple $$(n, q = 2^{64}, \sigma)$$ satisfies the target level of security $$\lambda_{target}$$.
    - Repeat for several values of $$n$$.

2. Model Generation for $$\lambda = \lambda_{target}$$.
    - At this point, we have several sets of points $$\{(n, q = 2^{64}, \sigma)\}$$ satisfying the target level of security $$\lambda_{target}$$. From here, we fit a model to this raw data ($$\sigma$$ as a function of $$n$$).

3. Model Verification.
    - For each model, we perform a verification check to ensure that the values output from the function $$\sigma(n)$$ provide the claimed level of security, $$\lambda_{target}$$.

These models are then used as input for Concrete, to ensure that the parameter space explored by the compiler attains the required security level. Note that we consider the `RC.BDGL16` lattice reduction cost model within the Lattice Estimator.
Therefore, when computing our security estimates, we use the call `LWE.estimate(params, red_cost_model = RC.BDGL16)` on the input parameter set `params`.

{% hint style="warning" %}
The cryptographic parameters are chosen considering the IND-CPA security model, and are selected with a bootstrapping failure probability fixed by the user. In particular, it is assumed that the results of decrypted computations are not shared by the secret key owner with any third parties, as such an action can lead to leakage of the secret encryption key. If you are designing an application where decryptions must be shared, you will need to craft custom encryption parameters which are chosen in consideration of the IND-CPA^D security model [1].

[1] Li, Baiyu, et al. “Securing approximate homomorphic encryption using differential privacy.” Annual International Cryptology Conference. Cham: Springer Nature Switzerland, 2022. https://eprint.iacr.org/2022/816.pdf
{% endhint %}

## Usage

To generate the raw data from the lattice estimator, use::

    make generate-curves

by default, this script will generate parameter curves for {80, 112, 128, 192} bits of security, using $$log_2(q) = 64$$.

To compare the current curves with the output of the lattice estimator, use:

    make compare-curves

this will compare the four curves generated above against the output of the version of the lattice estimator found in the [third_party folder](../../third_party).

To generate the associated cpp and rust code, use::

    make generate-code

further advanced options can be found inside the Makefile.

## Example

<!-- markdown-link-check-disable -->
To look at the raw data gathered in step 1., we can look in the [sage-object folder](../../tools/parameter-curves/sage-object). These objects can be loaded in the following way using SageMath:

    sage: X = load("128.sobj")

entries are tuples of the form: $$(n, log_2(q), log_2(\sigma), \lambda)$$. We can view individual entries via::

    sage: X["128"][0]
    (2366, 64.0, 4.0, 128.51)

To view the interpolated curves we load the `verified_curves.sobj` object inside the [sage-object folder](../../tools/parameter-curves/sage-object).

    sage: curves = load("verified_curves.sobj")

This object is a tuple containing the information required for the four security curves ({80, 112, 128, 192} bits of security). Looking at one of the entries:

    sage: curves[2][:3]
    (-0.026599462343105267, 2.981543184145991, 128)

Here we can see the linear model parameters $$(a = -0.026599462343105267, b = 2.981543184145991)$$ along with the security level 128. This linear model can be used to generate secure parameters in the following way: for $$q = 2^{64}$$, if we have an LWE dimension of $$n = 1536$$, then the required noise size is:

$$ \sigma = a * n + b = -37.85 $$

This value corresponds to the logarithm of the relative error size. Using the parameter set $$(n, log(q), \sigma = 2^{64 - 37.85})$$ in the Lattice Estimator confirms a 128-bit security level.
<!-- markdown-link-check-enable -->
