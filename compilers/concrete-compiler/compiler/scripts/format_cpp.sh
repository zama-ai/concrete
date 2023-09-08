#!/bin/bash

set -e -o pipefail

EXCLUDE_DIRS="-path ./include/boost-single-header -prune -o"

find ./{include,lib,src,tests} $EXCLUDE_DIRS -iregex '^.*\.\(cpp\|cc\|h\|hpp\)$' -print | xargs clang-format -i -style='file'

# show changes if any
git --no-pager diff --patience

# success if the diff is empty
git --no-pager diff --exit-code && exit 0

echo
echo "Formatting issue: Please run 'scripts/format_cpp.sh' in compilers/concrete-compiler/compiler and 'git add -p'"
exit 1
