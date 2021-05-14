Documentation README
======================

Documentation for the ``estimator`` is available `online <https://lwe-estimator.readthedocs.io/>`__.
This documentation can be generated locally by running the following code in the lwe-estimator directory:


::

    pipenv run make html

If documentation was previously generated locally, to ensure a full regeneration use:

::
    
    pipenv run make clean && rm -fr doc/_apidoc
    

