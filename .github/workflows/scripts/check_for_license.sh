#!/bin/bash

print_and_exit() {
    echo "Need to add license header to file $1"
    exit 1
}

EXCLUDE_DIRS="-path ./compiler/include/boost-single-header -prune -o"

files=$(find ./compiler/{include,lib,src} $EXCLUDE_DIRS -iregex '^.*\.\(cpp\|cc\|h\|hpp\)$' -print)

for file in $files
do
    cmp <(head -n 4 $file) <(echo "// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.") || print_and_exit $file
done

# Ignore python package namespace init file
files=$(find ./compiler/{include,lib,src} -iregex '^.*\.\(py\)$' ! -path ./compiler/lib/Bindings/Python/concrete/__init__.py)

for file in $files
do
    cmp <(head -n 2 $file) <(echo "#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.") || print_and_exit $file
done
