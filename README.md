<p align="center">
  <img width=170 height=170 src="logo.png">
  <h1 align="center">Concrete</h1>
</p>

The `concrete` ecosystem is a set of crates that implements Zama's variant of
[TFHE](https://eprint.iacr.org/2018/421.pdf). In a nutshell,
[fully homomorphic encryption (FHE)](https://en.wikipedia.org/wiki/Homomorphic_encryption), allows
you to perform computations over encrypted data, allowing you to implement Zero Trust services.

Concrete is based on the
[Learning With Errors (LWE)](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf) and the
[Ring Learning With Errors (RLWE)](https://eprint.iacr.org/2012/230.pdf) problems, which are well
studied cryptographic hardness assumptions believed to be secure even against quantum computers.

## Links

- [documentation](https://concrete.zama.ai)
- [whitepaper](http://whitepaper.zama.ai)

## Concrete crates

Concrete is implemented using the [Rust Programming language](https://www.rust-lang.org/), which
allows very fast, yet very secure implementations.

The ecosystem is composed of several crates (packages in the Rust language):

+ [`concrete`](concrete): A high-level library, useful to cryptographers that want to quickly
  implement homomorphic applications, without having to understand the details of the
  implementation.
+ [`concrete-core`](concrete-core): A low-level library, useful to cryptographers who want the
  fastest implementation possible, with all the settings at their disposal.
+ [`concrete-npe`](concrete-npe): A noise propagation estimator, used in `concrete` to simulate the
  evolution of the noise in ciphertexts, through homomorphic operations.
+ [`concrete-csprng`](concrete-csprng): A fast cryptographically secure pseudorandom number
  generator used in `concrete-core`.

## Installation

To use concrete, you will need the Rust compiler, and the FFTW library. The compiler can be
installed on linux and osx with the following command:

```bash
curl  --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Other rust installation methods are available on the
[rust website](https://forge.rust-lang.org/infra/other-installation-methods.html).

To install the FFTW library on MacOS, one could use the Homebrew package manager. To install
Homebrew, you can do the following:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

And then use it to install FFTW:

```bash
brew install fftw
```

**Note for Apple Silicon users**: Concrete is currently only available for x86 architecture.
To use it on Apple Silicon chip, you could use an x86_64 version (more detailed information
could be found [here](https://github.com/zama-ai/concrete/issues/65#issuecomment-902005481)).

To install FFTW on a debian-based distribution, you can use the following command:

```bash
sudo apt-get update && sudo apt-get install -y libfftw3-dev
```

# Credits

This library uses several dependencies and we would like to thank the contributors of those
libraries :

- [**FFTW**](https://crates.io/crates/fftw) (rust
  wrapper) : [Toshiki Teramura](https://github.com/termoshtt) (GPLv3)
- [**FFTW**](http://www.fftw.org) (lib) : [M. Frigo](http://www.fftw.org/~athena/)
  and [S. G. Johnson](http://math.mit.edu/~stevenj/) (GPLv3)
- [**itertools**](https://crates.io/crates/itertools): [bluss](https://github.com/bluss) (MIT /
  Apache 2.0)
- [**kolmogorov_smirnov**](https://crates.io/crates/kolmogorov_smirnov): [D. O. Crualaoich](https://github.com/daithiocrualaoich) (
  Apache 2.0)
- [**serde**](https://crates.io/crates/serde): [E. Tryzelaar](https://github.com/erickt)
  and [D. Tolnay](https://github.com/dtolnay) (MIT or Apache 2.0)
- [**colored**](https://crates.io/crates/colored): [T. Wickham](https://github.com/mackwic) (
  MPL-2.0)

We also use some crates published by `The Rust Project Developers` under the MIT or Apache 2.0
license :

- [**backtrace**](https://crates.io/crates/backtrace)
- [**rand**](https://crates.io/crates/rand)
- [**rand_distr**](https://crates.io/crates/rand_distr)
- [**num-traits**](https://crates.io/crates/num-traits)
- [**libc**](https://crates.io/crates/libc)
- [**num-integer**](https://crates.io/crates/num-integer)

We thank [Daniel May](https://gitlab.com/danieljrmay) for supporting this project and donating the
`concrete` crate.

# License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, 
please contact us at `hello@zama.ai`. 
