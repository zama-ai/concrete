FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y curl gcc

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install FFTW library
RUN apt-get install -y libfftw3-dev
