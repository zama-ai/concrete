# Documentation

## Setup doc environment

Create a local python virtual environment in the `docs` directory and install dependencies.

```shell
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

then in the `docs` directory, compile documentation:

```
$ make html
````

and open the html file `_build/html/index.html`

### Automatic rebuild on change

If you want to have an automatic preview on every changes you could install the `sphinx-autobuild`
python package and launch a local server, inside the `docs` directory do the following:

```shell
$ pip install sphinx-autobuild
$ python -m sphinx_autobuild . _build
```

The documentation is available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) by default.
