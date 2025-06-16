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

## Regular example

### Step 1: Define and Compile a Module in Python

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

### Step 2: Set Up the Rust Project

Initialize a new Rust project and add the required dependencies.

```shell
cargo init
cargo add concrete@=2.10.1-rc1 concrete-macro@=2.10.1-rc1
```

Place the `MyModule.zip` artifact in your project directory.

### Step 3: Import the Python-Compiled Module in Rust

Use the `concrete_macro::from_concrete_python_export_zip!` macro to import the module at build time.

```rust
mod my_module {
    use concrete_macro::from_concrete_python_export_zip;
    from_concrete_python_export_zip!("MyModule.zip");
}
```

This macro unpacks the artifact, triggers recompilation, reads metadata, and generates Rust APIs for the module's functions.

### Step 4: Use the Module in Rust

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

## TFHE-rs Ciphertext Interoperability

Starting from Concrete v2.11, you can define and use modules that operate directly on TFHE-rs ciphertexts, enabling seamless interoperability between Concrete and TFHE-rs in Rust.

### Step 1: Define and Compile a Module with TFHE-rs Types in Python

You can define a module in Python that uses TFHE-rs integer types as arguments and outputs. For example:

```python
from concrete import fhe
from concrete.fhe import tfhers

TFHERS_UINT_8_3_2_4096 = tfhers.TFHERSIntegerType(
    False,
    bit_width=8,
    carry_width=3,
    msg_width=2,
    params=tfhers.CryptoParams(
        lwe_dimension=909,
        glwe_dimension=1,
        polynomial_size=4096,
        pbs_base_log=15,
        pbs_level=2,
        lwe_noise_distribution=0,
        glwe_noise_distribution=2.168404344971009e-19,
        encryption_key_choice=tfhers.EncryptionKeyChoice.BIG,
    ),
)

@fhe.module()
class MyModule:

    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def my_func(x, y):
        x = tfhers.to_native(x)
        y = tfhers.to_native(y)
        return tfhers.from_native(x + y, TFHERS_UINT_8_3_2_4096)

def t(v):
    return tfhers.TFHERSInteger(TFHERS_UINT_8_3_2_4096, v)

inputset = [(t(0), t(0)), (t(2**6), t(2**6))]
my_module = MyModule.compile({"my_func": inputset})
my_module.server.save("test_tfhers.zip", via_mlir=True)
```

This produces a `test_tfhers.zip` artifact compatible with Rust and TFHE-rs.

### Step 2: Use the Module with TFHE-rs Ciphertexts in Rust

You can import and use the module in Rust, passing and receiving native TFHE-rs ciphertexts:

```rust
mod precompile {
    use concrete_macro::from_concrete_python_export_zip;
    from_concrete_python_export_zip!("src/test_tfhers.zip");
}

use tfhe::prelude::{FheDecrypt, FheEncrypt};
use tfhe::shortint::parameters::v0_10::classic::gaussian::p_fail_2_minus_64::ks_pbs::V0_10_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M64;
use tfhe::{generate_keys, FheUint8};

fn main() {
    // Key generation for TFHE-rs
    let config = tfhe::ConfigBuilder::with_custom_parameters(V0_10_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M64);
    let (client_key, _) = generate_keys(config);

    // Build Concrete keyset with TFHE-rs client key
    let mut secret_csprng = concrete::common::SecretCsprng::new(0u128);
    let mut encryption_csprng = concrete::common::EncryptionCsprng::new(0u128);
    let keyset = precompile::KeysetBuilder::new()
        .with_key_for_my_func_0_arg(&client_key)
        .generate(secret_csprng.pin_mut(), encryption_csprng.pin_mut());
    let server_keyset = keyset.get_server();

    // Encrypt inputs using TFHE-rs
    let arg_0 = FheUint8::encrypt(6u8, &client_key);
    let arg_1 = FheUint8::encrypt(4u8, &client_key);

    // Evaluate the Concrete circuit on TFHE-rs ciphertexts
    let mut server = precompile::server::my_func::ServerFunction::new();
    let output = server.invoke(&server_keyset, arg_0, arg_1);

    // Decrypt the result using TFHE-rs
    let decrypted: u8 = output.decrypt(&client_key);
    assert_eq!(decrypted, 10);
}
```

This workflow allows you to combine the high-level graph optimizations of Concrete with the operator-level flexibility of TFHE-rs, all within Rust.

## Notes

- The module must be compiled with `via_mlir=True` to be loaded in the Rust program.
- The Rust API is currently in beta and may evolve in future releases.
- The Python and Rust environments must use compatible versions of the Concrete toolchain.
- When using TFHE-rs ciphertext interoperability, ensure that the TFHE-rs client key used for encryption matches the one registered in the Concrete keyset for the corresponding argument.
