#!/bin/bash

set -e -o pipefail

echo "Formatting with " $(clang-format --version)

find ./{include,lib,src,tests} \
    \( -name "*.h" -or -name "*.hpp" -or -name "*.cpp" -or -name "*.cc" \) \
    -and -not -path "./include/boost-single-header*" \
    -and -not -path "./lib/Bindings/Rust*" \
    | xargs clang-format -i -style='file'

# show changes if any
git --no-pager diff --patience
# success if the diff is empty
git --no-pager diff --exit-code && exit 0

# write diff to file
git diff > format_cpp_diff.patch

echo
echo "Formatting issue: Please run 'scripts/format_cpp.sh' in compilers/concrete-compiler/compiler and 'git add -p'"
exit 1
