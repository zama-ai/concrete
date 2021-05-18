# Building the compiler

Generate the compiler build system, in the `build` directory

```sh
cmake -B build . -DLLVM_DIR=$LLVM_PROJECT/build/lib/cmake/llvm    -DMLIR_DIR=$LLVM_PROJECT/build/lib/cmake/mlir
```

Build the compiler

```sh
make -C build/ zamacompiler
```

Run the compiler

```sh
./build/src/zamacompiler
```