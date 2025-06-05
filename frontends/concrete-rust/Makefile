CARGO_SUBDIRS := concrete concrete-macro test concrete-keygen
COMPILER_BUILD_DIRECTORY := ../../../compilers/concrete-compiler/compiler/build

.PHONY: test clean run pcc

test:
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Running cargo test in $$dir..."; \
		COMPILER_BUILD_DIRECTORY=$(COMPILER_BUILD_DIRECTORY) cargo test --all-features --manifest-path $$dir/Cargo.toml -- --nocapture || exit 1; \
	done

clean:
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Cleaning target folder in $$dir..."; \
		rm -rf $$dir/target || exit 1; \
	done

clear_locks :
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Cleaning lock in $$dir..."; \
		rm -rf $$dir/Cargo.lock || exit 1; \
	done

run:
	cd test && \
		COMPILER_BUILD_DIRECTORY=$(COMPILER_BUILD_DIRECTORY) cargo run

pcc:
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Running pcc in $$dir..."; \
		COMPILER_BUILD_DIRECTORY=$(COMPILER_BUILD_DIRECTORY) cargo check --all-features --manifest-path $$dir/Cargo.toml || exit 1; \
	done

format:
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Running format in $$dir..."; \
		COMPILER_BUILD_DIRECTORY=$(COMPILER_BUILD_DIRECTORY) cargo fmt --manifest-path $$dir/Cargo.toml || exit 1; \
	done

regen_test_zips:
	cd test/src && \
	python ../python/test.py && \
	python ../python/test_tfhers.py
