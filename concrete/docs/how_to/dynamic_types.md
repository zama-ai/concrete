# Dynamic Types

Creating a dynamic type is done by using the `add_*` methods of the [ConfigBuilder](https://zama.ai).

| Kind      | Builder method                           |
| --------- | ---------------------------------------- |
| Booleans  | [add\_bool\_type](https://zama.ai)       |
| ShortInts | [add\_short\_int\_type](https://zama.ai) |
| Integers  | [add\_integer\_type](https://zama.ai)    |

These methods return an `encryptor`, which is the object you'll need to use to create values of your new type.

Types created dynamically still benefit from overloaded operators.

## Examples

Creating a 10-bit integer by combining five 2-bit ShortInts

```rust
// This requires integers feature enabled
#[cfg(feature = "integers")]
fn main() {
    use concrete::prelude::*;
    use concrete::{
        generate_keys, set_server_key, ConfigBuilder, RadixParameters,
        FheUint2Parameters,
    };

    let mut config = ConfigBuilder::all_disabled();
    let uint10_type = config.add_integer_type(RadixParameters {
        block_parameters: FheUint2Parameters::default().into(),
        num_block: 5,
    });

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let a = uint10_type.encrypt(177, &client_key);
    let b = uint10_type.encrypt(100, &client_key);

    let c: u64 = (a + b).decrypt(&client_key);
    assert_eq!(c, 277);
}
```

Another example using the CRT decomposition instead of the Radix one to represent 16-bit
integers. 

```rust
fn main() {
    use concrete::prelude::*;
    use concrete::{
        generate_keys, set_server_key, ConfigBuilder, CrtParameters,
        FheUint2Parameters,
    };
    let mut config = ConfigBuilder::all_disabled();
    let uint10_type = config.add_integer_type(CrtParameters {
        block_parameters: FheUint2Parameters::with_carry_2().into(),
        moduli: vec![7, 8, 9, 11, 13],
    });

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let a = uint10_type.encrypt(552, &client_key);
    let b = uint10_type.encrypt(1232, &client_key);

    let c: u64 = (a + b).decrypt(&client_key);
    assert_eq!(c, 552 + 1232);
}
```