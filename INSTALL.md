
# Installation

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

## macOS

To have the required C compiler and linker you'll either need to install the full __XCode__ IDE
(that you can install from the AppleStore) or install the __Xcode Command Line Tools__ by typing the
following command:

```bash
xcode-select --install
```

### Apple Silicon

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

## Linux

On linux, installing the required components depends on your distribution.
But for the typical Debian-based/Ubuntu-based distributions,
running the following command will install the needed packages:
```bash
sudo apt install build-essential
```

## Windows

Concrete is not currently supported natively on Windows but could be installed through the `WSL`.
