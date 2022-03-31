#!/bin/bash 

create_check_symlink () {
   ln -s $1 $2 -f
   if ! [ -e ${2} ] ; then
      echo "Broken link"
      echo $2
      exit 1
   fi
}

mkdir -p links_to_compiler_build/py/concretelang_core

cd links_to_compiler_build/py/concretelang_core

ln -s ../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/mlir -f

# Create directories needed for symlinks
mkdir -p concrete/lang/dialects
mkdir -p concrete/compiler
cd concrete
# Consider concrete as a package, as it's not detecting it as a namespace
touch __init__.py

py_prefix="$PWD/../../../../../compiler/build/tools/concretelang/python_packages/concretelang_core/concrete/"
pyfiles=`find $py_prefix -iname "*.py"`

for file in $pyfiles
do
    relative_path=`echo $file | sed s:$py_prefix::`
    create_check_symlink $file $relative_path
done

# Manually create dialect files
create_check_symlink ${py_prefix}lang/dialects/_FHE_ops_gen.py lang/dialects/fhe.py
create_check_symlink ${py_prefix}lang/dialects/_FHELinalg_ops_gen.py lang/dialects/fhelinalg.py
create_check_symlink ${py_prefix}lang/dialects/_ods_common.py lang/dialects/_ods_common.py
