import subprocess
from ctypes import *
import os
import numpy as np

v0_parameters_path = "cpp"

def compile():
    # Creating build directory
    try:
        os.mkdir(f"{v0_parameters_path}/build")
        print("> Successfully created build/ directory")
    except FileExistsError:
        print("> build/ directory already exists")
    # Compile the C++ source as a shared object
    subprocess.run(
        [
            "g++",
            "-c",
            "-o",
            f"{v0_parameters_path}/build/test.o",
            f"{v0_parameters_path}/test.cpp",
        ]
    )
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-o",
            f"{v0_parameters_path}/build/libtest.so",
            f"{v0_parameters_path}/build/test.o",
        ]
    )
    print("> Successfully compiled C++ source")


def load_library():
    # Load library in python and define argtype / restype
    lib = CDLL(f"{v0_parameters_path}/build/libtest.so")
    # defining the structure at python level
    class v0curves(Structure):
        _fields_ = [
            ("securityLevel", c_int),
            ("linearTerm1", c_double),
            ("linearTerm2", c_double),
            ("nAlpha", c_int),
            ("keyFormat", c_int),
        ]

    get = lib.security_estimator
    get.argtypes = [c_int, c_int]
    get.restype = POINTER(v0curves)
    print("> Successfully loading shared library")

    return get


def stringify_struct(struct):
    return f"security_level: {struct.contents.securityLevel}, linear_term1: {struct.contents.linearTerm1}, linear_term2: {struct.contents.linearTerm2} , nAlpha: {struct.contents.nAlpha}, keyFormat: {struct.contents.keyFormat} "

def check_codegen(
    curves_dict
):
    # compiling as shared  library
    compile()
    # loading library
    security_estimator = load_library()
    # checking everything

    for security_level, key_format in curves_dict:
        c_struct = security_estimator(security_level, key_format ) 
        python_struct = curves_dict[(security_level, key_format)]
        print(f"(securityLevel, keyFormat) = ({security_level, key_format} : {stringify_struct(c_struct)} ")
        assert python_struct[0] == c_struct.contents.linearTerm1, f"linearTerm1: (securityLevel, keyFormat) = ({security_level, key_format} -> (Py) {python_struct[0]} (C++) {c_struct.contents.linearTerm1})"
        assert python_struct[1] == c_struct.contents.linearTerm2, f"linearTerm2: (securityLevel, keyFormat) = ({security_level, key_format} -> (Py) {python_struct[1]} (C++) {c_struct.contents.linearTerm2})"
        assert python_struct[2] == c_struct.contents.nAlpha, f"nAlpha: (securityLevel, keyFormat) = ({security_level, key_format} -> (Py) {python_struct[2]} (C++) {c_struct.contents.nAlpha})"
    print(curves_dict)
    print("> Successfully compared C++ array with Python dictionary")


if __name__ == "__main__":
    from v0curves import curves_dict
    compile()
    load_library()
    check_codegen(curves_dict)
