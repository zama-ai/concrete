#!/bin/bash

mkdir -p links_to_compiler_build/md

cd links_to_compiler_build/md

yourfilenames=`find ../../../compiler/build/tools/concretelang/docs/concretelang -name "*.md"`

for entry in $yourfilenames
do
  ln -s "$entry" -f
done
