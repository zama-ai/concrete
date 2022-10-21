# Installation

{% hint style="warning" %}
Concrete 0.2.0 is the first version of the new Concrete Library. It is based on experimental features in `concrete-core` through the intermediate libraries (`concrete-integer`, `concrete-shortint`). It is published with a temporary dependency `concrete-core-experimental`. Future versions of Concrete 0.2 will be based on a public version of `concrete-core`.
{% endhint %}

## Importing into your project

To use `concrete` in your project, you first need to add it as a dependency in your `Cargo.toml`:

```toml
concrete = { version = "0.2.0", features = [ "booleans" ] }
```

## Choosing your features

`concrete` exposes different `cargo features` to customize the types and features used.

### Kinds.

`concrete` types

This crate exposes 3 kinds of data types. Each kind is enabled by activating its corresponding feature in the TOML line. Each kind may have multiple types:

| Kind      | Cargo Feature | Type(s)                                  |
| --------- | ------------- |------------------------------------------|
| Booleans  | `booleans`    | [FheBool]                                |
| ShortInts | `shortints`   | [FheUint2]<br>[FheUint3]<br>[FheUint4]   |
| Integers  | `integers`    | [FheUint8]<br>[FheUint12]<br>[FheUint16] |

[FheBool]: https://docs.rs/concrete/0.2.0/concrete/type.FheBool.html
[FheUint2]: https://docs.rs/concrete/0.2.0/concrete/type.FheUint2.html
[FheUint3]: https://docs.rs/concrete/0.2.0/concrete/type.FheUint3.html
[FheUint4]: https://docs.rs/concrete/0.2.0/concrete/type.FheUint4.html
[FheUint8]: https://docs.rs/concrete/0.2.0/concrete/type.FheUint8.html
[FheUint12]: https://docs.rs/concrete/0.2.0/concrete/type.FheUint12.html
[FheUint16]: https://docs.rs/concrete/0.2.0/concrete/type.FheUint16.html

### Serialization.

By enabling the `serde` feature, the different data types and keys exposed by the crate can be serialized / deserialized.

### Enabling all features.

Copy this if you would like to enable all features:

```toml
concrete = { version = "0.2.0", features = [ "booleans", "shortints", "integers", "serde"] }
```

***

## Supported platforms

As `concrete` relies on `concrete-core`, `concrete` is supported on:
  - `x86_64 Linux`
  - `aarch64 Linux`
  - `x86_64 macOS`
  - `aarch64 macOS`

Windows users can use `concrete` through the `WSL`.

### Working inside a cargo workspace

To be able to select the correct features depending on the target arch (x86_64 or aarch64),
concrete enables the option `resolver = "2"` by default.

However, when developing a crate inside a Cargo workspace,
this is no longer respected by cargo, and you will have to manually specify
this option in your `Cargo.toml` like so:

```toml
[workspace]
resolver = "2"
members = [
   "your-crate-that-uses-concrete", 
   "another-crate",
]
```

When compiling, if you see errors mentioning `aarch64` when you are on a `x86_64`
processor (or the opposite) then it is likely a `resolver` issue.

### Apple Silicon instructions

{% hint style="info" %}
macOS users who have Apple Silicon (`arm64`) devices can use `concrete` by compiling using the `nightly` toolchain
{% endhint %}

First, install the needed Rust toolchain:

```shell
rustup toolchain install nightly
```

Then, you can either:

* Manually specify the toolchain to use in each of the cargo commands:

For example:

```shell
cargo +nightly build
cargo +nightly test
```

* Or override the toolchain to use for the current project:

```shell
rustup override set nightly
# cargo will use the `nightly` toolchain.
cargo build
```

To check the toolchain that Cargo will use by default, you can use the following command:

```shell
rustup show
```
