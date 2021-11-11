import os
import subprocess

import pytest

from test_compiler_file_output.utils import assert_exists, content, remove, run

TEST_PATH = os.path.dirname(__file__)

CCOMPILER = 'cc'
ZAMACOMPILER = 'zamacompiler'

SOURCE_1 = f'{TEST_PATH}/return_13.ir'
SOURCE_2 = f'{TEST_PATH}/return_0.ir'
SOURCE_C_1 = f'{TEST_PATH}/main_return_13.c'
SOURCE_C_2 = f'{TEST_PATH}/main_return_0.c'
OUTPUT = f'{TEST_PATH}/output.mlir'
LIB = f'{TEST_PATH}/outlib'
LIB_STATIC = LIB + '.a'
LIB_DYNAMIC = LIB + '.so'
LIBS = (LIB_STATIC, LIB_DYNAMIC)

assert_exists(SOURCE_1, SOURCE_2, SOURCE_C_1, SOURCE_C_2)

def test_roundtrip():
    remove(OUTPUT)

    run(ZAMACOMPILER, SOURCE_1, '--action=roundtrip', '-o',  OUTPUT) 

    assert_exists(OUTPUT)
    assert content(SOURCE_1) == content(OUTPUT)

    remove(OUTPUT)


def test_roundtrip_many():
    remove(OUTPUT)

    run(ZAMACOMPILER, SOURCE_1, SOURCE_2, '--action=roundtrip', '-o',  OUTPUT)

    assert_exists(OUTPUT)
    assert f"{content(SOURCE_1)}{content(SOURCE_2)}" == content(OUTPUT)

    remove(OUTPUT)


def test_compile_library():
    remove(LIBS)

    run(ZAMACOMPILER, SOURCE_1, '--action=compile', '-o',  LIB)

    assert_exists(LIBS)

    EXE = './main.exe'
    remove(EXE)
    run(CCOMPILER, '-o', EXE, SOURCE_C_1, LIB_STATIC)

    result = subprocess.run([EXE], capture_output=True)
    assert 13 == result.returncode

    remove(EXE)
    run(CCOMPILER, '-o', EXE, SOURCE_C_1, LIB_DYNAMIC)

    result = subprocess.run([EXE], capture_output=True)
    assert 13 == result.returncode

    remove(LIBS, EXE)

def test_compile_many_library():
    remove(LIBS)

    run(ZAMACOMPILER, SOURCE_1, SOURCE_2, '--action=compile', '-o',  LIB)

    assert_exists(LIBS)

    EXE = './main.exe'
    remove(EXE)
    run(CCOMPILER, '-o', EXE, SOURCE_C_2, LIB_DYNAMIC)

    result = subprocess.run([EXE], capture_output=True)
    assert 0 == result.returncode

    remove(LIBS, EXE)
