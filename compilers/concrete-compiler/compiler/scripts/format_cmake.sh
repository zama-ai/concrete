#!/bin/bash

set -e -o pipefail

cmake-format -i CMakeLists.txt -c .cmake-format-config.py

find ./{include,lib,src,tests} -type f -name "CMakeLists.txt" | xargs -I % sh -c 'cmake-format -i % -c .cmake-format-config.py'

# show changes if any
git --no-pager diff --patience
# success if the diff is empty
git --no-pager diff --exit-code && exit 0

# write diff to file
git diff > format_cmake_diff.patch

echo
echo "Formatting issue: Please run 'scripts/format_cmake.sh' in compilers/concrete-compiler/compiler and 'git add -p'"
exit 1
