#!/bin/bash

set -o pipefail

find ./compiler/{include,lib,src}  \( -iname "*.h" -o -iname "*.cpp" -o -iname "*.cc" \) | xargs clang-format -i -style='file'
if [ $? -ne 0 ]
then
    exit 1
fi

# show changes if any
git diff
# fail if there is a diff, success otherwise
! ( git diff |  grep -q ^ ) || exit 1
