#!/usr/bin/env bash

# Fix OMP issues for macOS, https://github.com/zama-ai/concrete-ml-internal/issues/3951

set -e

UNAME=$(uname)
MACHINE=$(uname -m)
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
DO_REGENERATE=0

if [ "$UNAME" == "Darwin" ]
then

    # We need to source the venv here, since it's not done in the CI
    # shellcheck disable=SC1090,SC1091
    source  .venv/bin/activate

    # In the following, `command -v python` is for `which python` in a way which is approved by our
    # shell lint
    WHICH_VENV=$(command -v python | sed -e "s@bin/python@@")
    WHICH_PYTHON=$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')

    # To update the dylib changes (eg with major updates in torch), please uncomment this, and
    # then recopy to the given sections
    if [ $DO_REGENERATE -eq 1 ]
    then
        cd "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"

        LIST_OF_OMP_DYLIBS=$(find . -name "*omp*.dylib")

        for X in $LIST_OF_OMP_DYLIBS
        do
            if [[ "$X" != *"concrete"* ]]; then
                echo "rm \"\${WHICH_VENV}\"/lib/\"\${WHICH_PYTHON}\"/$X"
                echo "ln -s \"\${WHICH_VENV}\"/lib/\"\${WHICH_PYTHON}\"/site-packages/concrete/.dylibs/libomp.dylib \"\${WHICH_VENV}\"/lib/\"\${WHICH_PYTHON}\"/$X"
            fi
        done

        exit 255
    fi

    # The error is specific to python version and HW
    if [ "$MACHINE" == "arm64" ]
    then
        rm "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/./site-packages/xgboost/.dylibs/libomp.dylib
        ln -s "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/site-packages/concrete/.dylibs/libomp.dylib "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/./site-packages/xgboost/.dylibs/libomp.dylib
        rm "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/./site-packages/torch/lib/libomp.dylib
        ln -s "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/site-packages/concrete/.dylibs/libomp.dylib "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/./site-packages/torch/lib/libomp.dylib
        rm "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/./site-packages/sklearn/.dylibs/libomp.dylib
        ln -s "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/site-packages/concrete/.dylibs/libomp.dylib "${WHICH_VENV}"/lib/"${WHICH_PYTHON}"/./site-packages/sklearn/.dylibs/libomp.dylib
    else
        echo "Please have a look to libraries libiomp5.dylib related to torch and then"
        echo "apply appropriate fix"
        exit 255
    fi
fi
