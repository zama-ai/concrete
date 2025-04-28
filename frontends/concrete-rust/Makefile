CARGO_SUBDIRS := concrete concrete-macro test concrete-keygen

.PHONY: test clean
test:
	@for dir in $(CARGO_SUBDIRS); do \
	   echo "Running cargo test in $$dir..."; \
	   COMPILER_BUILD_DIRECTORY=../../../compilers/concrete-compiler/compiler/build cargo test --all-features --manifest-path $$dir/Cargo.toml -- --nocapture || exit 1; \
	done

clean:
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Cleaning target folder in $$dir..."; \
		rm -rf $$dir/target || exit 1; \
	done

run:
	cd test && \
 	COMPILER_BUILD_DIRECTORY=../../../compilers/concrete-compiler/compiler/build cargo run

format:
	@for dir in $(CARGO_SUBDIRS); do \
		echo "Running format in $$dir..."; \
		COMPILER_BUILD_DIRECTORY=$(COMPILER_BUILD_DIRECTORY) cargo fmt --manifest-path $$dir/Cargo.toml || exit 1; \
	done

regen_test_zips:
	python test/python/test.py
	python test/python/test_tfhers.py
