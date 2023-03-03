#!/bin/bash

set -e -o pipefail

cmake-format -i compiler/CMakeLists.txt -c compiler/.cmake-format-config.py

find ./compiler/{include,lib,src,tests} -type f -name "CMakeLists.txt" | xargs -I % sh -c 'cmake-format -i % -c compiler/.cmake-format-config.py'

# show changes if any
git --no-pager diff
# fail if there is a diff, success otherwise
git diff | ifne exit 1
