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
