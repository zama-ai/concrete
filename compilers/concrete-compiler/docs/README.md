# Bibliotech

Documentation is built with sphinx (https://www.sphinx-doc.org).

## Dependencies

In the docs repository (which contains this file):

It is recommended to first create a virtualenv:
```console
virtualenv .venv
source .venv/bin/activate
```

You need to install the Python dependencies: 
```console
pip install -r requirements.txt
pip install -r ../llvm-project/mlir/python/requirements.txt
```

You also need to install [Doxygen](https://www.doxygen.nl/index.html).

<!-- It is recommended to have `epstopdf` installed (https://www.ctan.org/pkg/epstopdf). Sphinx emits warnings without it.
On Ubuntu, it is part of the `texlive-font-utils` package. -->

## Build local docs


```console
make doc
```

To open the docs:

```console
xdg-open _build/html/index.html
```

## Automatic build on update

If you want the documentation to be updated as you are editing it, you can install an extra package called `sphinx-autobuild`
```console
pip install sphinx-autobuild
```

and launch (after a run of `make doc`) in the doc directory:

```console
python -m sphinx_autobuild . _build
```

This will launch a local web server on port 8000 with updated rendered version on each source modifications (but doesn't work on dialects changes, for which `make doc` need to be run again).
