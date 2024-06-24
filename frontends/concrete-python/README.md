# Python Frontend

## Installation for end-users

End-users should install `concrete-python` using `pip`:

```shell
pip install concrete-python
```

**Note:** Not all versions are available on PyPI. If you need a version that is not on PyPI (including nightly releases), you can install it from our package index by adding `--extra-index-url https://pypi.zama.ai/cpu/`. GPU wheels are also available under `https://pypi.zama.ai/gpu/` (check `https://pypi.zama.ai/` for all available platforms).

## Setup for development

Developers that want to contribute to the Concrete-Python project can use the following
approach to setup their environment.

```shell
# clone the repository
git clone https://github.com/zama-ai/concrete.git --recursive
cd concrete

# create virtual environment
cd frontends/concrete-python
make venv

# activate virtual environment
source .venv/bin/activate

# build the compiler bindings
cd ../../compilers/concrete-compiler/compiler
make python-bindings

# set bindings build directory as an environment variable
# *** NOTE ***: You must use the Release build of the compiler! 
# For now, the Debug build is not compatible with concrete-python
export COMPILER_BUILD_DIRECTORY=$(pwd)/build
echo "export COMPILER_BUILD_DIRECTORY=$(pwd)/build" >> ~/.bashrc

# run tests
cd ../../../frontends/concrete-python
make pytest
```

### VSCode setup

Alternatively you can use VSCode to develop Concrete-Python:

Suppose the compiler bindings were built in `/home/zama/concrete/compilers/concrete-compiler/compiler/build`:

- Create a `.env` file in the concrete-python root directory
- Determine the absolute path of the local compiler repository, e.g. `/home/zama/concrete`. Replace this with your 
path in the following two lines
- Add to it `PYTHONPATH=$(PYTHON_PATH):/home/zama/concrete/compilers/concrete-compiler/compiler/build/tools/concretelang/python_packages/concretelang_core/`
- Add to it `LD_PRELOAD=/home/zama/concrete/compilers/concrete-compiler/compiler/build/lib/libConcretelangRuntime.so`

You can now configure `pytest` in VScode and run the tests using the graphical interface.
