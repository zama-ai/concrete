#!/bin/bash

set +e

EXIT_CODE=0

for make_target in "$@"; do
    make --no-print-directory "${make_target}"
    if [[ "$?" != "0" ]]; then
        EXIT_CODE=1
    fi
done

exit "${EXIT_CODE}"
