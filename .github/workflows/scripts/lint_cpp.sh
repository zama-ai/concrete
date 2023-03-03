#!/bin/bash

print_usage() {
    local FD=$1
    echo "Usage: $0 [OPTION]" >&$FD
    echo "Check if the sources comply with the checks from .clang-tidy" >&$FD
    echo "" >&$FD
    echo "Options:" >&$FD
    echo "  -f, --fix                  Advise clang-tidy to fix any issue" >&$FD
    echo "                             found." >&$FD
    echo "  -h, --help                 Print this help." >&$FD
}

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

CLANG_TIDY_EXTRA_ARGS=()

# Parse arguments
while [ $# -gt 0 ]
do
    case $1 in
	-f|--fix)
	    CLANG_TIDY_EXTRA_ARGS+=("--fix")
	    ;;
	-h|--help)
	  print_usage 1
	  exit 0
	  ;;
	*)
	  print_usage 2
	  exit 1
	  ;;
    esac

    shift
done

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
    xargs clang-tidy -p . -header-filter="$TOP_SRCDIR/include/.*\.h" \
	  "${CLANG_TIDY_EXTRA_ARGS[@]}"
