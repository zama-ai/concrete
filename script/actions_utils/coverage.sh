#!/usr/bin/env bash

set -o pipefail
set +e

CURR_DIR=$(dirname "$0")

# Run diff-coverage
if [[ "$1" == "" ]]; then
    export BB="origin/main"
else
    export BB="origin/$1"
fi
make coverage | tee diff-coverage.txt

# Get exit code without closing the script
TEST_EXIT_CODE="$?"

# Format diff-coverage.txt for PR comment
poetry run python "$CURR_DIR"/coverage_report_format.py \
--diff-cover-exit-code "$TEST_EXIT_CODE" \
--diff-cover-output diff-coverage.txt

# Set exit code to the diff coverage check
exit "$TEST_EXIT_CODE"
