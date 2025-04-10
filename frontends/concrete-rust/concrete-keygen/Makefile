build:
	cargo build --release --features="cli"

build-webapp:
	wasm-pack build -d webapp/pkg --target web . --features="wasm"

# install wasm-pack first: cargo install wasm-pack
wasm-build-node:
	wasm-pack build --target nodejs . --features="wasm"

wasm-build-web:
	wasm-pack build --target web . --features="wasm"
