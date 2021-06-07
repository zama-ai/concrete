#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

check_buildfile() {
    local FILE="$1"

    [ -f "$FILE" ] ||
	die "$FILE not found. Please run this script from within your build " \
	    "directory."
}

check_buildfile "CMakeFiles/CMakeDirectoryInformation.cmake"
check_buildfile "compile_commands.json"

# Extract toplevel source directory from CMakeDirectoryInformation.cmake
# containing a line:
#
# set(CMAKE_RELATIVE_PATH_TOP_SOURCE "...")
TOP_SRCDIR=$(grep -o 'set\s*(\s*CMAKE_RELATIVE_PATH_TOP_SOURCE\s\+"[^"]\+")' \
  CMakeFiles/CMakeDirectoryInformation.cmake | \
  sed 's/set\s*(\s*CMAKE_RELATIVE_PATH_TOP_SOURCE\s\+"\([^"]\+\)")/\1/g')

[ $? -eq 0 -a ! -z "$TOP_SRCDIR" ] ||
    die "Could not extract CMAKE_RELATIVE_PATH_TOP_SOURCE from CMake files."

find "$TOP_SRCDIR/"{include,lib,src} \
     \( -iname "*.h" -o -iname "*.cpp" -o -iname "*.cc" \) | \
    xargs clang-tidy -p . -header-filter="$TOP_SRCDIR/include/.*\.h"
