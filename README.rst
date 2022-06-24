Parameter curves for Concrete
=============

This Github repository contains the code needed to generate the Parameter curves used inside Zama. The repository contains the following files:

- cpp/, Python scripts to generate a cpp file containing the parameter curves (needs updating)
- data/, a folder containing the data generated for previous curves. 
- estimator_new/, the Lattice estimator (TODO: add as a submodule and use dependabot to alert for new commits)
- old_files/, legacy files used for previous versions
- generate_data.py, functions to gather raw data from the lattice estimator
- verifiy_curves.py, functions to generate and verify curves from raw data

.. image:: logo.svg
   :align: center
   :width: 200


Example
-------------------
This is an example of how to generate the parameter curves, and save them to file.

::
    ./job.sh
::

This will generate several data files, {80, 96, 112, 128, 144, 160, 176, 192, 256}.sobj

To generate the parameter curves from the data files, we run 

`sage verify_curves.py`

this will generate a list of the form:

::

   [(-0.04042633119364589, 1.6609788641436722, 80, 'PASS', 450),
    (-0.03414780360867051, 2.017310258660345, 96, 'PASS', 450),
    (-0.029670137081135885, 2.162463714083856, 112, 'PASS', 450),
    (-0.02640502876522622, 2.4826422691043177, 128, 'PASS', 450),
    (-0.023821437305989134, 2.7177789440636673, 144, 'PASS', 450),
    (-0.02174358218716036, 2.938810548493322, 160, 'PASS', 498),
    (-0.019904056582117684, 2.8161252801542247, 176, 'PASS', 551),
    (-0.018610403247590085, 3.2996236848399008, 192, 'PASS', 606),
    (-0.014606812351714953, 3.8493629234693003, 256, 'PASS', 826)]
::

each element is a tuple (a, b, security, P, n_min), where (a,b) are the model 
parameters, security is the security level, P is a boolean value denoting PASS or 
FAIL of the verification, and n_min is the smallest reccomended value of `n` to be used.

Each model outputs a value of sigma, and is of the form:

`f(a, b, n) = max(ceil(a * n + b), -log2(q) + 2)`

where the -log2(q) + 2 term ensures that we are always using at least two bits of noise.

Version History
-------------------

Data for the curves are kept in /data. The following files are present:

::

    v0: generated using the {usvp, dual, decoding} attacks
    v0.1: generated using the {mitm, usvp, dual, decoding} attacks
    v0.2: generated using the lattice estimator
::

TODO List
-------------------

There are several updates which are still required.
    1. Consider Hybrid attacks (WIP, Michael + Ben are coding up hybrid-dual/hybrid-decoding estimates)
    2. CI/CD stuff for new pushes to the external LWE Estimator.
    3. Fully automate the process of finding n_{alpha} for each curve.
    4. Functionality for q =! 64? This is covered by the curve, but we currently don't account for it in the models, and it needs to be done manually.
    5. cpp file generation
