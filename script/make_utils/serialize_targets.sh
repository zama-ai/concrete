#!/bin/bash

set +e

EXIT_CODE=0

# Get the make executable
MAKE="$1"
shift

for make_target in "$@"; do
    "${MAKE}" --no-print-directory "${make_target}"
    if [[ "$?" != "0" ]]; then
        EXIT_CODE=1
    fi
done

exit "${EXIT_CODE}"
