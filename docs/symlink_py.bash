#!/bin/bash 

mkdir -p links_to_compiler_build/py/concretelang_core

cd links_to_compiler_build/py/concretelang_core

ln -s ../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/mlir -f

# Create directories needed for symlinks
mkdir -p concrete/lang/dialects
cd concrete
# Consider concrete as a package, as it's not detecting it as a namespace
touch __init__.py

py_prefix="$PWD/../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/"
pyfiles=`find $py_prefix -iname "*.py"`

for file in $pyfiles
do
    ln -s $file `echo $file | sed s:$py_prefix::` -f
done

# Manually create dialect files
ln -s ${py_prefix}lang/dialects/_FHE_ops_gen.py lang/dialects/fhe.py  -f
ln -s ${py_prefix}lang/dialects/_FHELinalg_ops_gen.py lang/dialects/fhelinalg.py  -f
ln -s ${py_prefix}lang/dialects/_ods_common.py lang/dialects/_ods_common.py  -f
