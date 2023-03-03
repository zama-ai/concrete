import os
import subprocess
import sys

import pytest

from test_compiler_file_output.utils import assert_exists, content, remove, run

TEST_PATH = os.path.dirname(__file__)

CCOMPILER = "cc"
CONCRETECOMPILER = "concretecompiler"

SOURCE_1 = f"{TEST_PATH}/return_13.ir"
SOURCE_2 = f"{TEST_PATH}/return_0.ir"
SOURCE_C_1 = f"{TEST_PATH}/main_return_13.c"
SOURCE_C_2 = f"{TEST_PATH}/main_return_0.c"
OUTPUT = f"{TEST_PATH}/output.mlir"
ARTIFACTS_DIR = f"{TEST_PATH}/outlib"
LIB_STATIC = ARTIFACTS_DIR + "/staticlib.a"
DYNAMIC_LIB_NAME = "/sharedlib.dylib" if sys.platform == "darwin" else "/sharedlib.so"
LIB_DYNAMIC = ARTIFACTS_DIR + DYNAMIC_LIB_NAME
LIBS = (LIB_STATIC, LIB_DYNAMIC)
HEADER_FILE = ARTIFACTS_DIR + "/fhecircuit-client.h"
CLIENT_PARAMS_FILE = ARTIFACTS_DIR + "/client_parameters.concrete.params.json"
ALL_ARTIFACTS = LIBS + (HEADER_FILE, CLIENT_PARAMS_FILE)

assert_exists(SOURCE_1, SOURCE_2, SOURCE_C_1, SOURCE_C_2)


def test_roundtrip():
    remove(OUTPUT)

    run(CONCRETECOMPILER, SOURCE_1, "--action=roundtrip", "-o", OUTPUT)

    assert_exists(OUTPUT)
    assert content(SOURCE_1) == content(OUTPUT)

    remove(OUTPUT)


def test_roundtrip_many():
    remove(OUTPUT)

    run(CONCRETECOMPILER, SOURCE_1, SOURCE_2, "--action=roundtrip", "-o", OUTPUT)

    assert_exists(OUTPUT)
    assert f"{content(SOURCE_1)}{content(SOURCE_2)}" == content(OUTPUT)

    remove(OUTPUT)


def test_compile_library():
    remove(ALL_ARTIFACTS)

    run(CONCRETECOMPILER, SOURCE_1, "--action=compile", "-o", ARTIFACTS_DIR)

    assert_exists(ALL_ARTIFACTS)

    EXE = "./main.exe"
    remove(EXE)
    run(CCOMPILER, "-o", EXE, SOURCE_C_1, LIB_STATIC)

    result = subprocess.run([EXE], capture_output=True)
    assert 13 == result.returncode

    remove(EXE)
    run(CCOMPILER, "-o", EXE, SOURCE_C_1, LIB_DYNAMIC)

    result = subprocess.run([EXE], capture_output=True)
    assert 13 == result.returncode

    remove(ALL_ARTIFACTS, EXE)


def test_compile_many_library():
    remove(ALL_ARTIFACTS)

    run(CONCRETECOMPILER, SOURCE_1, SOURCE_2, "--action=compile", "-o", ARTIFACTS_DIR)

    assert_exists(LIBS)

    EXE = "./main.exe"
    remove(EXE)
    run(CCOMPILER, "-o", EXE, SOURCE_C_2, LIB_DYNAMIC)

    result = subprocess.run([EXE], capture_output=True)
    assert 0 == result.returncode

    remove(ALL_ARTIFACTS, EXE)
