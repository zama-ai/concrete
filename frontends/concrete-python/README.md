# Python Frontend

## Setup for development

```shell
# clone the repository
git clone https://github.com/zama-ai/concrete.git
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
export COMPILER_BUILD_DIRECTORY=$(pwd)/build
echo "export COMPILER_BUILD_DIRECTORY=$(pwd)/build" >> ~/.bashrc

# run tests
cd ../../../frontends/concrete-python
make pytest
```
