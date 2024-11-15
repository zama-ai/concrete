#!/bin/bash -e
# Link all omp lib to concrete one to avoid load of different omp lib.

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
if [ $(basename "$SITE_PACKAGES") != "site-packages" ]; then
    echo "python site packages($SITE_PACKAGES) dirname is not equals to 'site-packages', you probably not execute this script in a venv"
    exit 1
fi

find "$SITE_PACKAGES" \( -not \( -path "$SITE_PACKAGES/concrete" -prune \) -name 'lib*omp5.dylib' -or -name 'lib*omp.dylib' \) -exec ln -f -s "$SITE_PACKAGES/concrete/.dylibs/libomp.dylib" {} \;
