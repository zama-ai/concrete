# Serialization of Ciphertexts and Keys

This document explains how to serialize and deserialize ciphertexts and secret keys when working with TFHE-rs in Rust.

Concrete already has its serilization functions (e.g. `tfhers_bridge.export_value`, `tfhers_bridge.import_value`, `tfhers_bridge.keygen_with_initial_keys`, `tfhers_bridge.serialize_input_secret_key`). However, when implementing a TFHE-rs computation in Rust, we must use a compatible serialization.

## Ciphertexts

We can deserialize `FheUint8` (and similarly other types) using `bincode`

```rust
use tfhe::FheUint8;

/// ...

fn load_fheuint8(path: &String) -> FheUint8 {
    let path_fheuint: &Path = Path::new(path);
    let serialized_fheuint = fs::read(path_fheuint).unwrap();
    let mut serialized_data = Cursor::new(serialized_fheuint);
    bincode::deserialize_from(&mut serialized_data).unwrap()
}
```

To serialize

```rust
fn save_fheuint8(fheuint: FheUint8, path: &String) {
    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &fheuint).unwrap();
    let path_ct: &Path = Path::new(path);
    fs::write(path_ct, serialized_ct).unwrap();
}
```

## Secret Key

We can deserialize `LweSecretKey` using `bincode`

```rust
use tfhe::core_crypto::prelude::LweSecretKey;

/// ...

fn load_lwe_sk(path: &String) -> LweSecretKey<Vec<u64>> {
    let path_sk: &Path = Path::new(path);
    let serialized_lwe_key = fs::read(path_sk).unwrap();
    let mut serialized_data = Cursor::new(serialized_lwe_key);
    bincode::deserialize_from(&mut serialized_data).unwrap()
}
```

To serialize

```rust
fn save_lwe_sk(lwe_sk: LweSecretKey<Vec<u64>>, path: &String) {
    let mut serialized_lwe_key = Vec::new();
    bincode::serialize_into(&mut serialized_lwe_key, &lwe_sk).unwrap();
    let path_sk: &Path = Path::new(path);
    fs::write(path_sk, serialized_lwe_key).unwrap();
}
```
