# Concrete Operates oN Ciphertexts Rapidly by Extending TfhE

Concrete is a [fully homomorphic encryption (FHE)](https://en.wikipedia.org/wiki/Homomorphic_encryption) library that implements Zama's variant of [TFHE](https://eprint.iacr.org/2018/421.pdf).
Concrete is based on the [Learning With Errors (LWE)](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf) problem and on the [Ring Learning With Errors (RLWE)](https://eprint.iacr.org/2012/230.pdf) problem, which are well studied cryptographic hardness assumptions believed to be secure even against quantum computers.

To use Concrete, you must install [Rust](https://www.rust-lang.org), [FFTW](http://www.fftw.org) and add [concrete](https://github.com/zama-ai/concrete) to the list of dependencies.

## Links

- [documentation](https://concrete.zama.ai)
- [whitepaper](http://whitepaper.zama.ai)

## Rust installation

To install rust on Linux or Macos, you can do the following:

```bash
curl  --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

If you want other rust installation methods, please refer to [rust website](https://forge.rust-lang.org/infra/other-installation-methods.html).

## FFTW and openssl installation

You also need to install FFTW and openssl library.

### MacOS

The more straightforward way to install fftw is to use Homebrew Formulae. To install homebrew, you can do the following:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

And then install FFTW and openssl.

```bash
brew install fftw
brew install openssl
```

### Linux

To install FFTW on a debian-based distribution, you can do the following:

```bash
sudo apt-get update && sudo apt-get install -y libfftw3-dev libssl-dev
```

### From source

If you want to install FFTW directly from source, you can follow the steps described in [FFTW's website](http://www.fftw.org/fftw2_doc/fftw_6.html).

## Use `concrete` in your own project

You need to **add the Concrete library as a dependency** in your Rust project.

To do so, simply add the dependency in the `Cargo.toml` file.
Here is a **simple example**:

```toml
[package]
name = "play_with_fhe"
version = "0.1.0"
authors = ["FHE Curious"]
edition = "2018"
# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
concrete_lib = "0.1"
```

Now, you can **compile** your project with the `cargo build` command, which will fetch the Concrete library.

It is also possible to build the library from source by cloning this repository and running:

```bash
cd concrete
make build
```

# Tests

To run the tests, you can do the following

```bash
cd concrete
make test
```

# Example

```rust
use concrete_lib::*;
use crypto_api::error::CryptoAPIError;

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

# Credits

This library uses several dependencies and we would like to thank the contributors of those libraries :

- [**FFTW**](https://crates.io/crates/fftw) (rust wrapper) : [Toshiki Teramura](https://github.com/termoshtt) (GPLv3)
- [**FFTW**](http://www.fftw.org) (lib) : [M. Frigo](http://www.fftw.org/~athena/) and [S. G. Johnson](http://math.mit.edu/~stevenj/) (GPLv3)
- [**itertools**](https://crates.io/crates/itertools): [bluss](https://github.com/bluss) (MIT / Apache 2.0)
- [**kolmogorov_smirnov**](https://crates.io/crates/kolmogorov_smirnov): [D. O. Crualaoich](https://github.com/daithiocrualaoich) (Apache 2.0)
- [**openssl**](https://crates.io/crates/openssl) (rust wrapper): [S. Fackler](https://github.com/sfackler) (Apache-2.0)
- [**openssl**](https://www.openssl.org) (lib):

- [**serde**](https://crates.io/crates/serde): [E. Tryzelaar](https://github.com/erickt) and [D. Tolnay](https://github.com/dtolnay) (MIT or Apache 2.0)
- [**colored**](https://crates.io/crates/colored): [T. Wickham](https://github.com/mackwic) (MPL-2.0)

We also use some crates published by `The Rust Project Developers` under the MIT or Apache 2.0 license :

- [**backtrace**](https://crates.io/crates/backtrace)
- [**rand**](https://crates.io/crates/rand)
- [**rand_distr**](https://crates.io/crates/rand_distr)
- [**num-traits**](https://crates.io/crates/num-traits)
- [**libc**](https://crates.io/crates/libc)
- [**num-integer**](https://crates.io/crates/num-integer)

# License

This software is distributed under the AGPL-v3 license. If you have any question, please contact us at hello@zama.ai.
