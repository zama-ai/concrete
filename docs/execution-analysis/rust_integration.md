# Rust Integration

This document explains how to use Fully Homomorphic Encryption (FHE) modules developed with **concrete-python** directly in Rust programs using the **Concrete** toolchain.

This workflow enables rapid prototyping in Python and seamless deployment in Rust, combining the flexibility of Python with the safety and performance of Rust.

## Overview

- Write and compile FHE modules in Python using `concrete-python`.
- Import the compiled module artifact into Rust using the `concrete-macro` crate.
- Use the generated Rust APIs for encryption, evaluation, and decryption.

## Prerequisites

- Python 3.8+
- Rust 1.70+
- `concrete-python` (>=2.10)
- `concrete` and `concrete-macro` Rust crates (>=2.10.1-rc1)

## Step 1: Define and Compile a Module in Python

Write your FHE logic in Python and compile it to an artifact compatible with the Rust toolchain. Here is an example of a small and simple module:

```python
from concrete import fhe

@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return (x + 1) % 256

    @fhe.function({"x": "encrypted"})
    def dec(x):
        return (x - 1) % 256

inputset = fhe.inputset(fhe.uint8)
module = MyModule.compile({"inc": inputset, "dec": inputset})

module.server.save(path="MyModule.zip", via_mlir=True)
```

This produces a `MyModule.zip` artifact containing the compiled FHE module.

## Step 2: Set Up the Rust Project

Initialize a new Rust project and add the required dependencies.

```shell
cargo init
cargo add concrete@=2.10.1-rc1 concrete-macro@=2.10.1-rc1
```

Place the `MyModule.zip` artifact in your project directory.

## Step 3: Import the Python-Compiled Module in Rust

Use the `concrete_macro::from_concrete_python_export_zip!` macro to import the module at build time.

```rust
mod my_module {
    use concrete_macro::from_concrete_python_export_zip;
    from_concrete_python_export_zip!("MyModule.zip");
}
```

This macro unpacks the artifact, triggers recompilation, reads metadata, and generates Rust APIs for the module's functions.

## Step 4: Use the Module in Rust

You can now use the FHE functions in Rust. The following example demonstrates a full FHE workflow:

```rust
use concrete::common::Tensor;

fn main() {
    // Prepare input and expected output tensors
    let input = Tensor::new(vec![5], vec![]);
    let expected_output = Tensor::new(vec![6], vec![]);

    // Key generation
    let mut secret_csprng = concrete::common::SecretCsprng::new(0u128);
    let mut encryption_csprng = concrete::common::EncryptionCsprng::new(0u128);
    let keyset = my_module::new_keyset(secret_csprng.pin_mut(), encryption_csprng.pin_mut());
    let client_keyset = keyset.get_client();

    // Create client stub for the 'inc' function
    let mut inc_client = my_module::client::inc::ClientFunction::new(&client_keyset, encryption_csprng);

    // Encrypt input and obtain evaluation keys
    let encrypted_input = inc_client.prepare_inputs(input);
    let evaluation_keys = keyset.get_server();

    // Create server stub for the 'inc' function
    let mut inc_server = my_module::server::inc::ServerFunction::new();

    // Evaluate the function on encrypted data
    let encrypted_output = inc_server.invoke(&evaluation_keys, encrypted_input);

    // Decrypt the output
    let decrypted_output = inc_client.process_outputs(encrypted_output);

    // Check correctness
    assert_eq!(decrypted_output.values(), expected_output.values());
}
```

## Notes

- The module must be compiled with `via_mlir=True` to be loaded in the Rust program.
- The Rust API is currently in beta and may evolve in future releases.
- The Python and Rust environments must use compatible versions of the Concrete toolchain.
