#!/usr/bin/env bash

# Testing Jupyter notebooks

set -e

LIST_OF_NOTEBOOKS=($(find examples -type f -name "*.ipynb" | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_checkpoints"))

# shellcheck disable=SC2068
for NOTEBOOK in ${LIST_OF_NOTEBOOKS[@]}
do
    echo
    echo
    echo "Refreshing ${NOTEBOOK}"

    START=$(date +%s)
    if jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}"; then
        STATUS="succeeded"
    else
        STATUS="failed"
    fi
    END=$(date +%s)
    TIME_EXEC=$((END-START))

    echo "Notebook ${NOTEBOOK} refresh took ${TIME_EXEC} seconds and ${STATUS}"
    echo
    echo

    if [ "${STATUS}" == "failed" ]
    then
        exit 255
    fi

done
