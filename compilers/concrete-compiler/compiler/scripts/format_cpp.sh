#!/bin/bash

set -e -o pipefail

EXCLUDE_DIRS="-path ./include/boost-single-header -prune -o"

find ./{include,lib,src,tests} $EXCLUDE_DIRS -iregex '^.*\.\(cpp\|cc\|h\|hpp\)$' -print | xargs clang-format -i -style='file'

# show changes if any
git --no-pager diff
# fail if there is a diff, success otherwise
git diff | ifne exit 1
