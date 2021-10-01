# Concrete Operates oN Ciphertexts Rapidly by Extending TfhE

Concrete is
a [fully homomorphic encryption (FHE)](https://en.wikipedia.org/wiki/Homomorphic_encryption) library
that implements Zama's variant of [TFHE](https://eprint.iacr.org/2018/421.pdf). Concrete is based on
the [Learning With Errors (LWE)](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf) problem and on
the [Ring Learning With Errors (RLWE)](https://eprint.iacr.org/2012/230.pdf) problem, which are well
studied cryptographic hardness assumptions believed to be secure even against quantum computers.

## Use `concrete` in your own project

You can use `cargo new play_with_fhe` to create a new project. You need to **add the Concrete
library as a dependency** in your Rust project.

To do so, simply add the dependency in the `Cargo.toml` file. Here is a **simple example**:

```toml
[package]
name = "play_with_fhe"
version = "0.1.0"
authors = ["FHE Curious"]
edition = "2018"
# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
concrete = "^0.1"
```

Now, you can **run** your project with the `RUSTFLAGS="-C target-cpu=native" cargo run --release`
command.

# Example

```rust
use concrete::*;

fn main() -> Result<(), CryptoAPIError> {
  // generate a secret key
  let secret_key = LWESecretKey::new(&LWE128_630);

  // the two values to add
  let m1 = 8.2;
  let m2 = 5.6;

  // specify the range and precision to encode messages into plaintexts
  // here we encode in [0, 10[ with 8 bits of precision and 1 bit of padding
  let encoder = Encoder::new(0., 10., 8, 1)?;

  // encode the messages into plaintexts
  let p1 = encoder.encode_single(m1)?;
  let p2 = encoder.encode_single(m2)?;

  // encrypt plaintexts
  let mut c1 = VectorLWE::encrypt(&secret_key, &p1)?;
  let c2 = VectorLWE::encrypt(&secret_key, &p2)?;

  // add the two ciphertexts homomorphically
  c1.add_with_padding_inplace(&c2)?;

  // decrypt and decode the result
  let m3 = c1.decrypt_decode(&secret_key)?;

  // print the result and compare to non-FHE addition
  println!("Real: {} + {} = {}", m1, m2, m1 + m2);
  println!(
    "FHE: {} + {} = {}",
    p1.decode()?[0],
    p2.decode()?[0],
    m3[0]
  );
  Ok(())
}
```
## Links

- [documentation](https://docs.zama.ai/concrete/lib)
- [TFHE](https://eprint.iacr.org/2018/421.pdf)


## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions,
please contact us at `hello@zama.ai`.
