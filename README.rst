Security Estimates for the Learning with Errors Problem
=======================================================

This `Sage <http://sagemath.org>`__ module provides functions for estimating the concrete security
of `Learning with Errors <https://en.wikipedia.org/wiki/Learning_with_errors>`__ instances.

The main intend of this estimator is to give designers an easy way to choose parameters resisting
known attacks and to enable cryptanalysts to compare their results and ideas with other techniques
known in the literature.

Usage Examples
--------------

::

    sage: load("estimator.py")
    sage: n, alpha, q = Param.Regev(128)
    sage: costs = estimate_lwe(n, alpha, q)
    usvp: rop:  ≈2^57.3,  red:  ≈2^57.3,  δ_0: 1.009214,  β:  101,  d:  349,  m:      220
     dec: rop:  ≈2^61.9,  m:      229,  red:  ≈2^61.9,  δ_0: 1.009595,  β:   93,  d:  357,  babai:  ≈2^46.8,  babai_op:  ≈2^61.9,  repeat:      293,  ε: 0.015625
    dual: rop:  ≈2^81.1,  m:      380,  red:  ≈2^81.1,  δ_0: 1.008631,  β:  115,  d:  380,  |v|:  688.951,  repeat:  ≈2^17.0,  ε: 0.007812

Online
------

You can `run the estimator
online <http://aleph.sagemath.org/?z=eJxNjcEKwjAQBe-F_kPoqYXYjZWkKHgQFPyLkOhii6mJyWrx782hiO84MPOcN9e6GohC2gHYkezrckdqfbzBZJwFN-MKE42TIR8hmhnOp8MRfqgNn6opiwdnxoXBcPZke9ZJxZlohRDbXknVSbGMMyXlpi-LhKTfGK1PWK-zr7O1NFHnz_ov2HwBPwsyhw==&lang=sage>`__
using the `Sage Math Cell <http://aleph.sagemath.org/>`__ server.

Coverage
--------

At present the following algorithms are covered by this estimator.

-  meet-in-the-middle exhaustive search
-  Coded-BKW [C:GuoJohSta15]
-  dual-lattice attack and small/sparse secret variant [EC:Albrecht17]
-  lattice-reduction + enumeration [RSA:LinPei11]
-  primal attack via uSVP [USENIX:ADPS16,ACISP:BaiGal14]
-  Arora-Ge algorithm [ICALP:AroGe11] using Gröbner bases
   [EPRINT:ACFP14]
  
The following distributions for the secret are supported:

- ``"normal"`` : normal form instances, i.e. the secret follows the noise distribution (alias: ``True``)
- ``"uniform"`` : uniform mod q (alias: ``False``)
- ``(a,b)`` : uniform in the interval ``[a,…,b]``
- ``((a,b), h)`` : exactly ``h`` components are ``∈ [a,…,b]\{0}``, all other components are zero

We note that distributions of the form ``(a,b)`` are assumed to be of fixed Hamming weight, with ``h = round((b-a)/(b-a+1) * n)``.

Above, we use `cryptobib <http://cryptobib.di.ens.fr>`__-style bibtex keys as references.

Documentation
-------------

Documentation for the ``estimator`` is available `here <https://lwe-estimator.readthedocs.io/>`__.

Evolution
---------

This code is evolving, new results are added and bugs are fixed. Hence, estimations from earlier
versions might not match current estimations. This is annoying but unavoidable at present. We
recommend to also state the commit that was used when referencing this project.

We also encourage authors to let us know if their paper uses this code. In particular, we thrive to
tag commits with those cryptobib ePrint references that use it. For example, `this commit
<https://bitbucket.org/malb/lwe-estimator/src/6295aa59048daa5d9598378386cb61887a1fe949/?at=EPRINT_Albrecht17>`__
corresponds to this `ePrint entry <https://ia.cr/2017/047>`__.

Contributions
-------------

Our intent is for this estimator to be maintained by the research community. For example, we
encourage algorithm designers to add their own algorithms to this estimator and we are happy to help
with that process.

More generally, all contributions such as bugfixes, documentation and tests are welcome. Please go
ahead and submit your pull requests. Also, don’t forget to add yourself to the list of contributors
below in your pull requests.

At present, this estimator is maintained by Martin Albrecht. Contributors are:

- Benjamin Curtis
- Cedric Lefebvre
- Fernando Virdia
- Florian Göpfert
- James Owen
- Léo Ducas
- Markus Schmidt
- Martin Albrecht
- Rachel Player
- Sam Scott

Please follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ in your submissions. You can use
`flake8 <http://flake8.pycqa.org/en/latest/>`__ to check for compliance. We use the following flake8
configuration (to allow longer line numbers and more complex functions):

::

    [flake8]
    max-line-length = 120
    max-complexity = 16
    ignore = E22,E241

Bugs
----

If you run into a bug, please open an `issue on bitbucket
<https://bitbucket.org/malb/lwe-estimator/issues?status=new&status=open>`__. Also, please check
first if the issue has already been reported.

Citing
------

If you use this estimator in your work, please cite

    | Martin R. Albrecht, Rachel Player and Sam Scott. *On the concrete hardness of Learning with Errors*.
    | Journal of Mathematical Cryptology. Volume 9, Issue 3, Pages 169–203, ISSN (Online) 1862-2984,
    | ISSN (Print) 1862-2976 DOI: 10.1515/jmc-2015-0016, October 2015

A pre-print is available as

    Cryptology ePrint Archive, Report 2015/046, 2015. https://eprint.iacr.org/2015/046

An updated version of the material covered in the above survey is available in
`Rachel Player's PhD thesis <https://pure.royalholloway.ac.uk/portal/files/29983580/2018playerrphd.pdf>`__.

License
-------

The esimator is licensed under the `LGPLv3+ <https://www.gnu.org/licenses/lgpl-3.0.en.html>`__ license.


Parameters from the Literature
------------------------------

The following estimates for various schemes from the literature illustrate the behaviour of the
``estimator``. These estimates do not necessarily correspond to the claimed security levels of the
respective schemes: for several parameter sets below the claimed security level by the designers’ is
lower than the complexity estimated by the ``estimator``. This is usually because the designers
anticipate potential future improvements to lattice-reduction algorithms and strategies. We
recommend to follow the designers’ decision. We intend to extend the estimator to cover these more
optimistic (from an attacker’s point of view) estimates in the future … pull requests welcome, as
always.

`New Hope <http://ia.cr/2015/1092>`__ ::

    sage: load("estimator.py")
    sage: n = 1024; q = 12289; stddev = sqrt(16/2); alpha = alphaf(sigmaf(stddev), q)
    sage: _ = estimate_lwe(n, alpha, q, reduction_cost_model=BKZ.sieve)
    usvp: rop: ≈2^313.1,  red: ≈2^313.1,  δ_0: 1.002094,  β:  968,  d: 2096,  m:     1071
     dec: rop: ≈2^410.0,  m:     1308,  red: ≈2^410.0,  δ_0: 1.001763,  β: 1213,  d: 2332,  babai: ≈2^395.5,  babai_op: ≈2^410.6,  repeat:  ≈2^25.2,  ε: ≈2^-23.0
    dual: rop: ≈2^355.5,  m:     1239,  red: ≈2^355.5,  δ_0: 1.001884,  β: 1113,  repeat: ≈2^307.0,  d: 2263,  c:        1

`Frodo <http://ia.cr/2016/659>`__ ::

    sage: load("estimator.py")
    sage: n = 752; q = 2^15; stddev = sqrt(1.75); alpha = alphaf(sigmaf(stddev), q)
    sage: _ = estimate_lwe(n, alpha, q, reduction_cost_model=BKZ.sieve)
    usvp: rop: ≈2^173.0,  red: ≈2^173.0,  δ_0: 1.003453,  β:  490,  d: 1448,  m:      695
     dec: rop: ≈2^208.3,  m:      829,  red: ≈2^208.3,  δ_0: 1.003064,  β:  579,  d: 1581,  babai: ≈2^194.5,  babai_op: ≈2^209.6,  repeat:      588,  ε: 0.007812
    dual: rop: ≈2^196.2,  m:      836,  red: ≈2^196.2,  δ_0: 1.003104,  β:  569,  repeat: ≈2^135.0,  d: 1588,  c:        1

`TESLA <http://ia.cr/2015/755>`__ ::

    sage: load("estimator.py")
    sage: n = 804;  q = 2^31 - 19; alpha = sqrt(2*pi)*57/q; m = 4972
    sage: _ = estimate_lwe(n, alpha, q, m=m, reduction_cost_model=BKZ.sieve)
    usvp: rop: ≈2^129.3,  red: ≈2^129.3,  δ_0: 1.004461,  β:  339,  d: 1937,  m:     1132
     dec: rop: ≈2^144.9,  m:     1237,  red: ≈2^144.9,  δ_0: 1.004148,  β:  378,  d: 2041,  babai: ≈2^130.9,  babai_op: ≈2^146.0,  repeat:       17,  ε: 0.250000
    dual: rop: ≈2^139.4,  m:     1231,  red: ≈2^139.4,  δ_0: 1.004180,  β:  373,  repeat:  ≈2^93.0,  d: 2035,  c:        1

`SEAL <https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/>`__ ::

    sage: load("estimator.py")
    sage: n = 2048; q = 2^54 - 2^24 + 1; alpha = 8/q; m = 2*n
    sage: _ = estimate_lwe(n, alpha, q, secret_distribution=(-1,1), reduction_cost_model=BKZ.sieve, m=m)
    Warning: the LWE secret is assumed to have Hamming weight 1365.
    usvp: rop: ≈2^129.7,  red: ≈2^129.7,  δ_0: 1.004479,  β:  337,  d: 3914,  m:     1865,  repeat:        1,  k:        0,  postprocess:        0
     dec: rop: ≈2^144.4,  m:  ≈2^11.1,  red: ≈2^144.4,  δ_0: 1.004154,  β:  377,  d: 4272,  babai: ≈2^131.2,  babai_op: ≈2^146.3,  repeat:        7,  ε: 0.500000
    dual: rop: ≈2^134.2,  m:  ≈2^11.0,  red: ≈2^134.2,  δ_0: 1.004353,  β:  352,  repeat:  ≈2^59.6,  d: 4091,  c:    3.909,  k:       32,  postprocess:       10

`LightSaber <https://www.esat.kuleuven.be/cosic/pqcrypto/saber/files/SABER_KEM_Round_2.zip>`__ ::

     sage: load("estimator.py")
     sage: n = 512
     sage: q = 8192
     sage: alpha_0 = alphaf(sqrt(10/4.0), q, sigma_is_stddev=True)  # error
     sage: alpha_1 = alphaf(sqrt(21/4.0), q, sigma_is_stddev=True)  # secret
     sage: primal_usvp(n, alpha_0, q, secret_distribution=alpha_1, m=n, reduction_cost_model=BKZ.ADPS16)  # not enough samples
     Traceback (most recent call last):
     ...
     NotImplementedError: secret size 0.000701 > error size 0.000484

     sage: primal_usvp(n, alpha_1, q, secret_distribution=alpha_0, m=n, reduction_cost_model=BKZ.ADPS16)
             rop:  2^118.0
             red:  2^118.0
         delta_0: 1.003955
            beta:      404
               d:     1022
               m:      509
