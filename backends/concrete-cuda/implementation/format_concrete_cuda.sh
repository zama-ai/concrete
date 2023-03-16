#!/bin/bash

find ./{include,src,test} -iregex '^.*\.\(cpp\|cu\|h\|cuh\)$' -print | xargs clang-format-11 -i -style='file'
cmake-format -i CMakeLists.txt -c ../../../compilers/concrete-compiler/compiler/.cmake-format-config.py

find ./{include,src,test} -type f -name "CMakeLists.txt" | xargs -I % sh -c 'cmake-format -i % -c ../../../compilers/concrete-compiler/compiler/.cmake-format-config.py'

