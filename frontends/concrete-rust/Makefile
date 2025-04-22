CARGO_SUBDIRS := concrete concrete-macro test concrete-keygen

.PHONY: test
test:
	@for dir in $(CARGO_SUBDIRS); do \
	   echo "Running cargo test in $$dir..."; \
	   COMPILER_BUILD_DIRECTORY=../../../compilers/concrete-compiler/compiler/build cargo test --manifest-path $$dir/Cargo.toml || exit 1; \
	done
