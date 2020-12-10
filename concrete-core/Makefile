build:
	RUSTFLAGS="-C target-cpu=native" cargo build --release
test:
	RUSTFLAGS="-C target-cpu=native" cargo test --release -- crypto_api
doc:
	RUSTDOCFLAGS="--html-in-header katex-header.html" cargo +nightly doc --no-deps --open
