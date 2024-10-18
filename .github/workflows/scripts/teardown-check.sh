#!/bin/bash -e

grep setup-instance -Rl .github/workflows/ | xargs grep -L teardown-instance &> missing-teardown.txt

if [ -s missing-teardown.txt ]; then
    echo "There are missing teardown-instance jobs in following jobs:"
    echo
    cat missing-teardown.txt
    exit 1
fi
