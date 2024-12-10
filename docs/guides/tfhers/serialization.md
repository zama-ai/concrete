# Serialization of Ciphertexts and Keys

This document explains how to serialize and deserialize ciphertexts and secret keys when working with TFHE-rs in Rust.

Concrete already has its serilization functions (e.g. `tfhers_bridge.export_value`, `tfhers_bridge.import_value`, `tfhers_bridge.keygen_with_initial_keys`, `tfhers_bridge.serialize_input_secret_key`). However, when implementing a TFHE-rs computation in Rust, we must use a compatible serialization.

## Ciphertexts

We should deserialize `FheUint8` using safe serialization functions from TFHE-rs

```rust
use tfhe::FheUint8;
use tfhe::safe_serialization::{safe_deserialize, safe_serialize};

const SERIALIZE_SIZE_LIMIT: u64 = 1_000_000_000;

/// ...

fn load_fheuint8(path: &String) -> FheUint8 {
    let file = fs::File::open(path).unwrap();
    safe_deserialize(file, SERIALIZE_SIZE_LIMIT).unwrap()
}
```

To serialize

```rust
fn save_fheuint8(fheuint: FheUint8, path: &String) {
    let file = fs::File::create(path).unwrap();
    safe_serialize(fheuint, file, SERIALIZE_SIZE_LIMIT).unwrap()
}
```

## Secret Key

We should deserialize `LweSecretKey` using safe serialization functions from TFHE-rs

```rust
use tfhe::core_crypto::prelude::LweSecretKey;

/// ...

fn load_lwe_sk(path: &String) -> LweSecretKey<Vec<u64>> {
    let file = fs::File::open(path).unwrap();
    safe_deserialize(file, SERIALIZE_SIZE_LIMIT).unwrap()
}
```

To serialize

```rust
fn save_lwe_sk(lwe_sk: LweSecretKey<Vec<u64>>, path: &String) {
    let file = fs::File::create(path).unwrap();
    safe_serialize(lwe_sk, file, SERIALIZE_SIZE_LIMIT).unwrap()
}
```
