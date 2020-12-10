build:
	RUSTFLAGS="-C target-cpu=native" cargo build --release
test:
	RUSTFLAGS="-C target-cpu=native" cargo test --release
test-doc:
	RUSTFLAGS="-C target-cpu=native" cargo test --release --doc
doc:
	RUSTFLAGS="-C target-cpu=native" cargo doc --release --open
