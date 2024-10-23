# TFHE-rs / Concrete-ML interoperability tutorial

This is a work in progress tutorial, to describe how to deal with TFHE-rs and Concrete ML interoperability.

## Principle

The principle is to have a single Rust file, where we have ciphertexts, and to call some Python files to simulate what happens on the server side with Concrete (and soon Concrete ML). For now, there are two calls to Python:
- the first one to have the generation of the evaluation key in the Concrete format, from the TFHE-rs secret key; this represents something done on the client side
- the second one to have the actual FHE computation, with Concrete and only with public information

## Workflow

- Step 1: clean everything

```
rustup update
cargo clean
rm -rf server_dir client_dir
mkdir server_dir client_dir

rm -rf .venv
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Step 2: compile and run the Rust program

```
rm -f server_dir/*.txt client_dir/*.txt; cargo run --release
```

