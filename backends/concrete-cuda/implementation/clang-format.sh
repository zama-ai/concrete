#!/bin/bash

find ./{include,src,test} -iregex '^.*\.\(cpp\|cu\|h\|cuh\)$' -print | xargs clang-format-11 -i -style='file'

