#!/bin/bash

mkdir -p links_to_compiler_build/py/zamalang_core

cd links_to_compiler_build/py/zamalang_core

ln -s ../../../../compiler/build/tools/zamalang/python_packages/zamalang_core/mlir

mkdir zamalang
cd zamalang

ln -s ../../../../../compiler/build/tools/zamalang/python_packages/zamalang_core/zamalang/dialects/_HLFHE_ops_gen.py hlfhe.py  -f
ln -s ../../../../../compiler/build/tools/zamalang/python_packages/zamalang_core/zamalang/dialects/_HLFHELinalg_ops_gen.py hlfhelinalg.py  -f
ln -s ../../../../../compiler/build/tools/zamalang/python_packages/zamalang_core/zamalang/dialects/_ods_common.py _ods_common.py  -f

ln -s ../../../../../compiler/build/tools/zamalang/python_packages/zamalang_core/zamalang/compiler.py compiler.py  -f

touch __init__.py