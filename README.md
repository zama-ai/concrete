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

- [documentation](https://docs.zama.ai/concrete/lib)
- [whitepaper](http://whitepaper.zama.ai)
- [community website](https://community.zama.ai)

## Concrete crates

Concrete is implemented using the [Rust Programming language](https://www.rust-lang.org/), which
allows very fast, yet very secure implementations.

The ecosystem is composed of several crates (packages in the Rust language):

+ [`concrete`](concrete): A high-level library, useful to cryptographers that want to quickly
  implement homomorphic applications, without having to understand the details of the
  implementation.
+ [`concrete-core`](concrete-core): A low-level library, useful to cryptographers who want the
  fastest implementation possible, with all the settings at their disposal.
+ [`concrete-boolean`](concrete-boolean): A high-level library, implementing homomorphic Boolean 
  gates, making it easy to run any kind of circuits over encrypted data.
+ [`concrete-npe`](concrete-npe): A noise propagation estimator, used in `concrete` to simulate the
  evolution of the noise in ciphertexts, through homomorphic operations.
+ [`concrete-csprng`](concrete-csprng): A fast cryptographically secure pseudorandom number
  generator used in `concrete-core`.
+ [`concrete-commons`](concrete-commons): contains types and traits to manipulate objects in a 
  consistent way throughout the ecosystem.

## Installation

To use concrete, you will need the following things:
- A Rust compiler
- A C compiler & linker
- make

The Rust compiler can be installed on __Linux__ and __macOS__ with the following command:

```bash
curl  --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Other rust installation methods are available on the
[rust website](https://forge.rust-lang.org/infra/other-installation-methods.html).

### macOS

To have the required C compiler and linker you'll either need to install the full __XCode__ IDE
(that you can install from the AppleStore) or install the __Xcode Command Line Tools__ by typing the
following command:

```bash
xcode-select --install
```

#### Apple Silicon

Concrete currently __only__ supports the __x86_64__ architecture.
You can however, use it on Apple Silicon chip thanks to `Rosetta`
at the expense of slower execution times.

To do so, you need to compile `concrete` for the `x86_64` architecture
and let Rosetta2 handle the conversion, we do that by using the `x86_64` toolchain.

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

- Or override the toolchain to use:
```console
rustup override set stable-x86_64-apple-darwin
# cargo will use the `stable-x86_64-apple-darwin` toolchain.
cargo build
```

### Linux

On linux, installing the required components depends on your distribution.
But for the typical Debian-based/Ubuntu-based distributions,
running the following command will install the needed packages:
```bash
sudo apt install build-essential
```

### Windows

Concrete __does not__ work natively on Windows. You can however, if you feel adventurous, use the `WSL`.

One thing to note when using concrete on WSL is that to improve compile times,
the concrete repository should be located in the WSL's own space (that is, not somewhere under `/mnt/c`).
For example in your WSL's user home directory.

## Credits

This library uses several dependencies and we would like to thank the contributors of those
libraries.

We thank [Daniel May](https://gitlab.com/danieljrmay) for supporting this project and donating the
`concrete` crate.

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions,
please contact us at `hello@zama.ai`.
