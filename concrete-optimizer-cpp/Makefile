# dev or release
CARGO_PROFILE ?= release
# debug or release
PROFILE_SUBDIR ?= release

ROOT = ..
TARGET_DIR = $(ROOT)/target
PROFILE_DIR = $(TARGET_DIR)/$(PROFILE_SUBDIR)
CXXBRIDGE_DIR = $(TARGET_DIR)/cxxbridge/concrete-optimizer-cpp/src

INTERFACE_LIB = $(TARGET_DIR)/libconcrete_optimizer_cpp.a
INTERFACE_HEADER = src/cpp/concrete-optimizer.hpp
INTERFACE_CPP =  src/cpp/concrete-optimizer.cpp
INTERFACE_SOURCES = $(INTERFACE_HEADER) $(INTERFACE_CPP)

SOURCES = $(shell find $(ROOT)/concrete-optimizer/src) \
	$(shell find $(ROOT)/concrete-optimizer-cpp/src -name '*.rs')
TESTS_SOURCES = tests/src/main.cpp
TEST_DEP_LIBS = -l pthread -ldl

$(INTERFACE_SOURCES) $(INTERFACE_LIB): $(SOURCES)
	cd $(ROOT) && cargo build -p concrete-optimizer-cpp --profile $(CARGO_PROFILE)
	cp $(CXXBRIDGE_DIR)/concrete-optimizer.rs.h  $(INTERFACE_HEADER)
	cp $(CXXBRIDGE_DIR)/concrete-optimizer.rs.cc $(INTERFACE_CPP)
	cp $(PROFILE_DIR)/libconcrete_optimizer_cpp.a $(TARGET_DIR)

tests/tests_exe: $(INTERFACE_SOURCES) $(INTERFACE_LIB) $(TESTS_SOURCES)
	g++ -o $@ $(TESTS_SOURCES) $(INTERFACE_CPP) $(INTERFACE_LIB) -I $(shell dirname $(INTERFACE_HEADER)) $(TEST_DEP_LIBS)
	chmod +x $@

build: $(INTERFACE_SOURCES) $(INTERFACE_LIB)

test: tests/tests_exe
	./tests/tests_exe

test-ci:
	$(MAKE) CARGO_PROFILE=dev PROFILE_SUBDIR=debug test
	git diff --check src/cpp || echo Please commit the new version of generated files

clean:
	rm -f $(INTERFACE_LIB) tests/tests_exe
