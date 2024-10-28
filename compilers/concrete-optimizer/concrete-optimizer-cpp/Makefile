# dev or release
CARGO_PROFILE ?= release
# debug or release
PROFILE_SUBDIR ?= release

MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT = $(realpath $(MAKEFILE_DIR)/..)
CXXBRIDGE_DIR = $(ROOT)/target/cxxbridge/concrete-optimizer-cpp/src

# In target intermediate files
INTERFACE_LIB_ORIG = $(ROOT)/target/$(PROFILE_SUBDIR)/libconcrete_optimizer_cpp.a
INTERFACE_HEADER_ORIG = $(CXXBRIDGE_DIR)/concrete-optimizer.rs.h
INTERFACE_CPP_ORIG = $(CXXBRIDGE_DIR)/concrete-optimizer.rs.cc

# Final files
INTERFACE_LIB = $(ROOT)/target/libconcrete_optimizer_cpp.a
INTERFACE_HEADER = src/cpp/concrete-optimizer.hpp
INTERFACE_CPP =  src/cpp/concrete-optimizer.cpp

SOURCES = $(shell find $(ROOT)/concrete-optimizer/src) \
	$(shell find $(ROOT)/concrete-optimizer-cpp/src -name '*.rs')

build: $(INTERFACE_LIB) $(INTERFACE_CPP) $(INTERFACE_HEADER)

$(INTERFACE_LIB_ORIG) $(INTERFACE_HEADER_ORIG) $(INTERFACE_CPP_ORIG): $(SOURCES)
	cd $(ROOT) && cargo build -p concrete-optimizer-cpp --profile $(CARGO_PROFILE)

$(INTERFACE_HEADER): $(INTERFACE_HEADER_ORIG)
	cp $< $@

$(INTERFACE_CPP): $(INTERFACE_CPP_ORIG)
	cp $< $@

$(INTERFACE_LIB): $(INTERFACE_LIB_ORIG)
	cp $< $@

TESTS_SOURCES = tests/src/main.cpp
TEST_DEP_LIBS = -l pthread -ldl
tests/tests_exe: $(INTERFACE_LIB) $(INTERFACE_HEADER) $(INTERFACE_CPP) $(TESTS_SOURCES)
	g++ -Wall -Werror -Wextra -std=c++17 -o $@ $(TESTS_SOURCES) $(INTERFACE_CPP) $(INTERFACE_LIB) -I $(shell dirname $(INTERFACE_HEADER)) $(TEST_DEP_LIBS)
	chmod +x $@

test: tests/tests_exe
	./tests/tests_exe

test-ci:
	# force regeneration of sources
	rm -rf $(INTERFACE_HEADER) $(INTERFACE_CPP) $(INTERFACE_HEADER_ORIG) $(INTERFACE_CPP_ORIG)
	$(MAKE) CARGO_PROFILE=dev PROFILE_SUBDIR=debug test
	git diff --exit-code src/cpp || (echo Please commit the new version of generated files && exit 1)
