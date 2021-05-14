#!/usr/bin/env bash
###############################################################################
#                              Run Sage doctests
###############################################################################
SAGE_ROOT=$(sage -c "import os; print(os.environ['SAGE_ROOT'])")
export SAGE_ROOT="$SAGE_ROOT"

# shellcheck source=/dev/null
source "$SAGE_ROOT/local/bin/sage-env"
for file in "$@"; do
    PYTHONIOENCODING=UTF-8 PYTHONPATH=$(pwd) sage-runtests "$file"
done
