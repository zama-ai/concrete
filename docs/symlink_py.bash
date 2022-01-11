#!/bin/bash 

mkdir -p links_to_compiler_build/py/concretelang_core

cd links_to_compiler_build/py/concretelang_core

ln -s ../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/mlir -f

mkdir -p concrete/lang

cd concrete

ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/lang/dialects/_FHE_ops_gen.py lang/fhe.py  -f
ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/lang/dialects/_FHELinalg_ops_gen.py lang/fhelinalg.py  -f
ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/lang/dialects/_ods_common.py lang/_ods_common.py  -f

ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/compiler.py compiler.py  -f

touch lang/__init__.py __init__.py
