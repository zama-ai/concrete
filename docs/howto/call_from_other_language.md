# Calling from other languages

After doing a compilation, we endup with a couple of artifacts, including crypto parameters and a binary file containing the executable circuit. In order to be able to encrypt and run the circuit properly, we need to know how to interpret these artifacts, and there are a couple of utility functions to load them. These utility functions can be accessed through a variety of languages, including Python, Cpp, and Rust. [The Rust bindings](https://github.com/zama-ai/concrete/tree/release/2.4.x/compilers/concrete-compiler/compiler/lib/Bindings/Rust) (built on top of the [CAPI](https://github.com/zama-ai/concrete/tree/release/2.4.x/compilers/concrete-compiler/compiler/include/concretelang-c)) can be a good example for someone who wants to build bindings for another language.

## Calling from Rust

`bindgen` is used to generate Rust FFI bindings to the CAPI
[The Rust bindings](https://github.com/zama-ai/concrete/tree/release/2.4.x/compilers/concrete-compiler/compiler/lib/Bindings/Rust) are built on top of the CAPI in order to provide a safer, and more Rusty API. Although you can use `bindgen` (as we did to build the Rust bindings) to generate the Rust FFI from the CAPI and use it as is, we will here show how to use the Rust API that is built on top of that, as it's easier to use.

![](../\_static/calling\_from\_other\_lang\_rust\_bindings.jpg)


### Demo

We will use a really simple example for a demo, but the same steps can be done for any other circuit. `example.mlir` will contain the MLIR below:

```mlir
func.func @main(%arg0: tensor<4x4x!FHE.eint<6>>, %arg1: tensor<4x2xi7>) -> tensor<4x2x!FHE.eint<6>> {
   %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<4x4x!FHE.eint<6>>, tensor<4x2xi7>) -> (tensor<4x2x!FHE.eint<6>>)
   %tlu = arith.constant dense<[40, 13, 20, 62, 47, 41, 46, 30, 59, 58, 17, 4, 34, 44, 49, 5, 10, 63, 18, 21, 33, 45, 7, 14, 24, 53, 56, 3, 22, 29, 1, 39, 48, 32, 38, 28, 15, 12, 52, 35, 42, 11, 6, 43, 0, 16, 27, 9, 31, 51, 36, 37, 55, 57, 54, 2, 8, 25, 50, 23, 61, 60, 26, 19]> : tensor<64xi64>
   %result = "FHELinalg.apply_lookup_table"(%0, %tlu): (tensor<4x2x!FHE.eint<6>>, tensor<64xi64>) -> (tensor<4x2x!FHE.eint<6>>)
   return %result: tensor<4x2x!FHE.eint<6>>
}
```

You can use the `concretecompiler` binary to compile this MLIR program. Same can be done with `concrete-python`, as we only need the compilation artifacts at the end.

```bash
$ concretecompiler --action=compile -o rust-demo example.mlir
```

You should be able to see artifacts listed in the `rust-demo` directory

```bash
$ ls rust-demo/
client_parameters.concrete.params.json  compilation_feedback.json  fhecircuit-client.h  sharedlib.so  staticlib.a
```

Now we want to use the Rust bindings in order to call the compiled circuit.

```rust
use concrete_compiler::compiler::{KeySet, LambdaArgument, LibrarySupport};
```

The main `struct` to manage compilation artifacts is `LibrarySypport`. You will have to create one with the path you used during compilation, then load the result of the compilation

```rust
let lib_support = LibrarySupport::new(
        "/path/to/your/rust-demo/",
        None,
    )
    .unwrap();
let compilation_result = lib_support.load_compilation_result().unwrap();
```

Using the compilation result, you can load the server lambda (the entrypoint to the executable compiled circuit) as well as the client parameters (containing crypto parameters)

```rust
let server_lambda = lib_support.load_server_lambda(&compilation_result).unwrap();
let client_params = lib_support.load_client_parameters(&compilation_result).unwrap();
```

The client parameters will serve the client to generate keys and encrypt arguments for the circuit

```rust
let key_set = KeySet::new(&client_params, None, None, None).unwrap();
let args = [
        LambdaArgument::from_tensor_u8(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], &[4, 4])
            .unwrap(),
        LambdaArgument::from_tensor_u8(&[1, 2, 1, 2, 1, 2, 1, 2], &[4, 2]).unwrap(),
    ];
let encrypted_args = key_set.encrypt_args(&args).unwrap();
```

Only evaluation keys are required for the execution of the circuit. You can execute the circuit on the encrypted arguments via `server_lambda_call`

```rust
let eval_keys = key_set.evaluation_keys().unwrap();
let encrypted_result = lib_support
        .server_lambda_call(&server_lambda, &encrypted_args, &eval_keys)
        .unwrap()
```

At this point you have the encrypted result and can decrypt it using the keyset which holds the secret key

```rust
let result_arg = key_set.decrypt_result(&encrypted_result).unwrap();
println!("result tensor dims: {:?}", result_arg.dims().unwrap());
println!("result tensor data: {:?}", result_arg.data().unwrap());
```

There is also a couple of tests in [compiler.rs](https://github.com/zama-ai/concrete/blob/release/2.4.x/compilers/concrete-compiler/compiler/lib/Bindings/Rust/src/compiler.rs) that can show how to both compile and run a circuit between a client and server using serialization.
