# Concrete Operates oN Ciphertexts Rapidly by Extending TfhE

concrete is a Rust crate (library) meant to abstract away the details of Fully Homomorphic Encryption (FHE) to enable
non-cryptographers to build applications that use FHE.

FHE is a powerful cryptographic tool, which allows computation to be performed directly on encrypted data without
needing to decrypt it first.

## Example

```rust
use concrete::{ConfigBuilder, generate_keys, set_server_key, FheUint8};
use concrete::prelude::*;

fn main() {
    let config = ConfigBuilder::all_disabled()
        .enable_default_uint8()
        .build();

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let clear_a = 27u8;
    let clear_b = 128u8;

    let a = FheUint8::encrypt(clear_a, &client_key);
    let b = FheUint8::encrypt(clear_b, &client_key);

    let result = a + b;

    let decrypted_result: u8 = result.decrypt(&client_key);

    let clear_result = clear_a + clear_b;

    assert_eq!(decrypted_result, clear_result);
}
```

## Links

- [documentation](https://docs.zama.ai/concrete/lib)
- [TFHE](https://eprint.iacr.org/2018/421.pdf)

## License

This software is distributed under the **BSD-3-Clause-Clear** license with an 
exemption that gives rights to use our patents for research,
evaluation and prototyping purposes, as well as for your personal projects. 

If you want to use Concrete in a commercial product however,
you will need to purchase a separate commercial licence.

If you have any questions, please contact us at `hello@zama.ai`.