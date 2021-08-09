Parameter curves for Concrete
=============

This Github repository contains the code needed to generate the Parameter curves used inside Zama. The repository contains the following files:

- estimator.py, a copy of the LWE Estimator 
- scripts.py, a copy of all scripts required to generate the parameter curves
- data/, a folder containing the data generated for previous curves. This folder currently contains "v0.sobj", i.e. the data used for the v0 curves

Example
-------------------
This is an example of how to generate the parameter curves using the v0.sobj data file.


::

    sage: load("scripts.py")
    sage: interps = []
    sage: results = load("v0.sobj")
    sage: for result in results:
    sage:     interps.append(interpolate_result(result, log_q = 64))
    sage: interps
    [(-0.040476778656126484, 1.143346508563902),
     (-0.03417207792207793, 1.4805194805194737),
     (-0.029681716023268107, 1.752723426758335),
     (-0.0263748887657055, 2.0121439233304894),
     (-0.023730136557783763, 2.1537066948924095),
     (-0.021604493958972515, 2.2696862472846204),
     (-0.019897520946588438, 2.4423829771964796),
     (-0.018504919354426233, 2.6634073426215745),
     (-0.017254242957361113, 2.7353702447139026),
     (-0.016178309410530816, 2.8493969373734758),
     (-0.01541034709414119, 3.1982749283836283),
     (-0.014327640360322604, 2.899270827311096)]
::

Finding the value of n_{alpha} is done manually. We can also verify the interpolants which are generated at the same time:

::

    # verify the interpolant used for lambda = 256 (which is interps[-1])
    sage: z = verify_interpolants(interps[-1], (128,2048), 64)
    [... code runs, can take ~10 mins ...]
    # find the index corresponding to n_alpha, which is where security drops below the target security level (256 here)
    sage: n_alpha = find_nalpha(z, 256)
    653
    
    # so the model in this case is 
    (-0.014327640360322604, 2.899270827311096, 653)
    # which corresponds to
    # sd(n) = max(-0.014327640360322604 * n +  2.899270827311096, -logq + 2), n >= 653
::

TODO List
-------------------

There are several updates which are still required.
    1. Consider Hybrid attacks (WIP, Michael + Ben are coding up hybrid-dual/hybrid-decoding estimates)
    2. Consider MITM attacks for small values of n (WIP, Michael has coded this up in the Zama internal estimator, I just need to add it to the parameter curves code).
    3. As part of (2), the version of the LWE Estimator used needs to be edited to use Zama's internal estimator.
    4. Add an example of how to _generate_ the parameter curves in the first place.
    5. CI/CD stuff for new pushes to the external LWE Estimator.
    6. Fully automate the process of finding n_{alpha} for each curve.
    7. Functionality for q =! 64? This is covered by the curve, but we currently don't account for it in the models, and it needs to be done manually.
