"""
Setup concrete namespace.
"""

import warnings

# Do not modify, this is to have a compatible namespace package
# https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#pkg-resources-style-namespace-packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    __import__("pkg_resources").declare_namespace(__name__)  # pragma: no cover
