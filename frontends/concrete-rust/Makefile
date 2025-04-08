build:
	cargo build --release --features="cli"

# install wasm-pack first: cargo install wasm-pack
wasm-build-node:
	wasm-pack build --target nodejs . --features="wasm"

wasm-build-web:
	wasm-pack build --target web . --features="wasm"

