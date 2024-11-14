#!/bin/sh

# This file tests that the example is working

shell_blocks=$(sed -n '/^```sh/,/^```/ p' < README.md | sed '/^```sh/d' | sed '/^```/d')

set -e
output=$(eval "$shell_blocks" 2>&1) || echo "$output"
