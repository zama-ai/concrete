#!/usr/bin/env bash

set -o pipefail
set +e

CURR_DIR=$(dirname "$0")

# Format diff-coverage.txt for PR comment
poetry run python "$CURR_DIR"/coverage_report_format.py \
global-coverage \
--global-coverage-json-file "$1" \
--global-coverage-output-file diff-coverage.txt
