//! Part 1 of the Concrete guide where you set everything up to use Concrete library
//!
//! # Step 1: Install Rust and FFTW
//!
//!
//! To install rust on Linux or Macos, you can do the following
//!
//! ```bash
//! curl  --tlsv1.2 -sSf https://sh.rustup.rs | sh
//! ```
//!
//! If you want other rust installation methods, please refer to [rust website](https://forge.rust-lang.org/infra/other-installation-methods.html)
//!
//! To use Concrete, you also need to install FFTW library.
//!
//! ## Macos
//!
//! The more straightward way to install fftw is to use Homebrew Formulae. To install homebrew, you can do the following
//!
//! ```bash
//! /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
//! ```
//!
//! And then install FFTW
//!
//! ```bash
//! brew install fftw
//! ```

//! ## Linux
//!
//! To install FFTW on a debian-based distribution, you can do :
//!
//! ```bash
//! sudo apt-get update && sudo apt-get install -y libfftw3-dev
//! ```
//!
//! ## From source
//!
//! If you want to install FFTW from source, you can follow the steps described in [FFTW's website](http://www.fftw.org/fftw2_doc/fftw_6.html)
//!
//! # Step 2: Create a new Rust project (if you don't already have one)
//!
//! To create a new **Rust project**, you need to have Rust installed so you can run this command:
//! ```bash
//! cargo new play_with_fhe
//! ```
//!
//! # Step 3: Add Concrete to your dependencies
//!
//! First thing first, you need to **add the Concrete library as a dependency** in your Rust project.
//! To do so, simply write ``concrete = "0.1.0"`` in the ``Cargo.toml`` file.
//!
//! Here is a **simple example**:
//!
//! ```toml
//! [package]
//! name = "play_with_fhe"
//! version = "0.1.0"
//! authors = ["FHE Curious"]
//! edition = "2018"
//!
//! # See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html
//!
//! [dependencies]
//! concrete = "0.1"
//! itertools = "0.9.0"
//! ```
//! We also added ``itertools`` because we will use it in the next parts of the guide.
//!
//! Now, you can **compile** your project with the ``cargo build`` command, which will fetch the Concrete library.
//!
//! # Step 4: Import the main API module
//!
//! The last thing to do in order to use the Concrete library is to **import the ``crypto_api`` module**.
//! We can do that by writing ``use concrete;``
//!
//! As an **example** you can use the your empty project from the previous step, here is the file ``main.rs``:
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     println!("Hello, world!");
//! }
//! ```
//!
//! And now you can compile and run your code with
//! ```bash
//! cargo run
//! ```
//!
//! You're all set for the [next part](super::guide_part2_encrypt_decrypt)!
