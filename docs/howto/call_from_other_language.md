# Calling from other languages

After doing a compilation, we end up with a couple of artifacts, including crypto parameters and a binary file containing the executable circuit. In order to be able to encrypt and run the circuit properly, we need to know how to interpret these artifacts, and there are a couple of utility functions which can be used to load them. These utility functions can be accessed through a variety of languages, including Python and C++.

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
$ concretecompiler --action=compile -o python-demo example.mlir
```

You should be able to see artifacts listed in the `python-demo` directory

```bash
$ ls python-demo/
client_parameters.concrete.params.json  compilation_feedback.json  fhecircuit-client.h  sharedlib.so  staticlib.a
```

Now we want to use the Python bindings in order to call the compiled circuit.

```python
from concrete.compiler import (ClientSupport, LambdaArgument, LibrarySupport)
```

The main `struct` to manage compilation artifacts is `LibrarySupport`. You will have to create one with the path you used during compilation, then load the result of the compilation

```python
lib_support = LibrarySupport.new("/path/to/your/python-demo/")
compilation_result = lib_support.reload()
```

Using the compilation result, you can load the server lambda (the entrypoint to the executable compiled circuit) as well as the client parameters (containing crypto parameters)

```python
server_lambda = lib_support.load_server_lambda(compilation_result)
client_params = lib_support.load_client_parameters(compilation_result)
```

The client parameters will serve the client to generate keys and encrypt arguments for the circuit

```python
client_support = ClientSupport.new()
key_set = client_support.key_set(client_params)
args = [
	LambdaArgument.from_tensor_u8([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [4, 4]),
	LambdaArgument.from_tensor_u8([1, 2, 1, 2, 1, 2, 1, 2], [4, 2])
]
encrypted_args = client_support.encrypt_arguments(client_params, key_set, args)
```

Only evaluation keys are required for the execution of the circuit. You can execute the circuit on the encrypted arguments via `server_lambda_call`

```python
eval_keys = key_set.get_evaluation_keys()
encrypted_result = lib_support.server_call(server_lambda, encrypted_args, eval_keys)
```

At this point you have the encrypted result and can decrypt it using the keyset which holds the secret key

```python
result_arg = client_support.decrypt_result(client_params, key_set, encrypted_result)
print("result tensor dims: {}".format(result_arg.n_values()))
print("result tensor data: {}".format(result_arg.get_values()))
```

There is also a couple of tests in [test_compilation.py](https://github.com/zama-ai/concrete/blob/main/compilers/concrete-compiler/compiler/tests/python/test_compilation.py) that can show how to both compile and run a circuit between a client and server using serialization.
