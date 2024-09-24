#!/usr/bin/env bash

set -e

LOG_FILE=$(mktemp /tmp/actionlint.script.XXXXXX)
SUMMARY_LOG_FILE=$(mktemp /tmp/actionlint.script.XXXXXX)

# Get actionlint errors
actionlint | cat > "$LOG_FILE"

# Get only where the errors are, not their type
grep -v .yml "$LOG_FILE" | grep -v ^"    |" | cat > "$SUMMARY_LOG_FILE"

# Check errors which are not whitelisted
if python3 scripts/actionlint/actionlint_check_with_whitelists.py < "$SUMMARY_LOG_FILE";
then
    echo "Successful end"
    exit 0
else
    echo "Full log file: "
    cat "$LOG_FILE"

    echo
    echo "Summary log file:"
    cat "$SUMMARY_LOG_FILE"
    exit 255
fi

