#!/bin/bash

set -e -o pipefail

find ./compiler/{include,lib,src} -iregex '^.*\.\(cpp\|cc\|h\|hpp\)$' | xargs clang-format -i -style='file'

# show changes if any
git --no-pager diff
# fail if there is a diff, success otherwise
git diff | ifne exit 1
