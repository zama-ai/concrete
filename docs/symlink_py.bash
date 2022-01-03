#!/bin/bash

mkdir -p links_to_compiler_build/py/concretelang_core

cd links_to_compiler_build/py/concretelang_core

ln -s ../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/mlir

mkdir -p concretelang

cd concretelang

ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/lang/dialects/_FHE_ops_gen.py fhe.py  -f
ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/lang/dialects/_FHELinalg_ops_gen.py fhelinalg.py  -f
ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/lang/dialects/_ods_common.py _ods_common.py  -f

ln -s ../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/compiler.py compiler.py  -f

touch __init__.py