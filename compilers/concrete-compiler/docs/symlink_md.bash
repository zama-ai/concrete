#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "You must provide the compiler build directory"
    exit 1
fi

mkdir -p links_to_compiler_build/md

cd links_to_compiler_build/md

yourfilenames=`find $1/tools/concretelang/docs/concretelang -name "*.md"`

for entry in $yourfilenames
do
  ln -s "$entry" -f
done
