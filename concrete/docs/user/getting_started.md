# Getting Started


## What is concrete ?


``concrete`` is a Rust crate (library) meant to abstract away the details of
FHE (Fully Homomorphic Encryption) to enable non cryptographers to build
applications that use FHE.

If you do not know what problems FHE solves, we suggest taking a look at our 
[6 minute introduction to homomorphic encryption]

This crate provides different types which are the counterparts of native Rust type
(such as ``bool``, ``u8``, ``u16``) in FHE domain.

[6 minute introduction to homomorphic encryption]: https://6min.zama.ai/


## Importing in your project

### Supported platforms

As `concrete` relies on `concrete-core`, `concrete` is only supported on `x86_64 Linux` 
and `x86_64 macOS`.

Windows user can use `concrete` through the `WSL`.

macOS users which have the newer M1 (`arm64`) devices can use `concrete` by cross-compiling to
`x86_64` and run their program with rosetta.

First install the needed rust toolchain:

```console
# Install the macOS x86_64 toolchain (you only need to do this once)
rustup toolchain install --force-non-host stable-x86_64-apple-darwin
```

Then you can either:

- Manually specify the toolchain to use in each of the cargo commands:

For example:

```console
cargo +stable-x86_64-apple-darwin build
cargo +stable-x86_64-apple-darwin test
```

- Or override the toolchain to use for the current project:

```console
rustup override set stable-x86_64-apple-darwin
# cargo will use the `stable-x86_64-apple-darwin` toolchain.
cargo build
```


### Cargo.toml Import

To be able to use ``concrete`` in your project, you need to add it as a dependency in your
``Cargo.toml``:

```toml
concrete = { version = "0.1.0", features = [ "booleans" ] }
```

### 

(choosing-your-features)=
### Choosing your features

#### Kinds

``concrete`` types

This crate exposes 3 kinds of data types, each kind is enabled by activating its corresponding
feature in the toml line. Each kind may have multiple types:

| Kind      | Cargo Feature | Type(s)                                  |
|-----------|---------------|------------------------------------------|
| Booleans  | `booleans`    | [FheBool]                                |
| ShortInts | `shortints`   | [FheUint2]<br>[FheUint3]                 |
| Integers  | `integers`    | [FheUint8]<br>[FheUint12]<br>[FheUint16] |


#### Serialization

By enabling the `serde` feature, the different data types and keys exposed by the crate can be
serialized / deserialized.

#### Dynamic types

When you enable the feature tied to a `type kind`, the crate will expose some predefined types that you can
configure and use. However, some `type kind` like the `integers` are a bit more flexible, and you may wish to
create your own types (based on a `kind`) at runtime.

See our how-to {ref}`how-to-create-dynamic-types`.


## A Note On Performances 

Due to their nature, FHE types are slower than native types, so it is recommended to always build and run your
project in release mode (`cargo build --release`, `cargo run --release`).

If you ever need to use a debug a debugger, you can tell cargo to generate debug info by adding in your toml
```toml
[profile.release]
debug = true
```

Another option that _may_ improve performances is to enable `fat` link time optimizations:
```toml
[profile.release]
lto = "fat"
```

You should compare the run time with and without lto to see if it improves performances.

% TODO update links

[FheBool]: https://google.com
[FheUint2]: https://google.com
[FheUint3]: https://google.com
[FheUint8]: https://google.com
[FheUint12]: https://google.com
[FheUint16]: https://google.com
