#!/bin/sh

# This file tests that the example is working

shell_blocks=$(sed -n '/^```sh/,/^```/ p' < README.md | sed '/^```sh/d' | sed '/^```/d')

set -e
output=$(eval "$shell_blocks" 2>&1) || echo "$output"

result=$(echo "$output" | grep "result: " | sed 's/result: //g')

expected="31"
if [ $result -eq $expected ]
then
    exit 0
else
    echo "expected result to be $expected, but result was $result"
    exit 1
fi
