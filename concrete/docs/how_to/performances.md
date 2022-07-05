# Performances

## Compile and run in `release` mode

Due to their nature, FHE types are slower than native types, so it is recommended to always build and run your project in release mode (`cargo build --release`, `cargo run --release`).

## Link time optimization

Another option that _may_ improve performances is to enable `fat` link time optimizations:

```toml
[profile.release]
lto = "fat"
```

You should compare the run time with and without LTO to see if it improves performances.

## Choose the best-suited type(s)

With FHE types, the more precision you have, the more the computations are expensive.
Therefore, it is important to choose the smallest type that can represent all your values.

`concrete` gives the ability to create `dynamic` types, that is, types that are created and customized
at runtime to better fit your needs and try to gain performances.
This feature can be a great option if for example, you only need 10 bits of precision
and that `concrete` does not expose an integer type with exactly 10 bits.

You can see our [how to](../how_to/dynamic_types.md).
