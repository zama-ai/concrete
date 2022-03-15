# Building the compiler

Install MLIR following https://mlir.llvm.org/getting_started/
Use commit c2415d67a564

Install pybind11:

```sh
pip install pybind11
```

Build concrete library:

```sh
git clone https://github.com/zama-ai/concrete_internal
cd concrete_internal
git checkout engine_c_api
cd concrete-ffi
RUSTFLAGS="-C target-cpu=native" cargo build --release 
```

Generate the compiler build system, in the `build` directory

```sh
export LLVM_PROJECT="PATH_TO_LLVM_PROJECT"
export CONCRETE_PROJECT="PATH_TO_CONCRETE_INTERNAL_PROJECT"
make build-initialized
```

Build the compiler

```sh
make concretecompiler
```

Test the compiler

```sh
#TODO: let cmake set this PATH
export LD_LIBRARY_PATH="path_to_concrete-compiler/compiler/build/lib/Runtime/"
make test
```

Run the compiler

```sh
./build/src/concretecompiler
```
