#!/bin/bash

VENV=$(mktemp -d)/.venv

python3 -m venv $VENV
source $VENV/bin/activate

pip install -r requirements.txt --quiet --disable-pip-version-check
pip install pip-licenses --quiet --disable-pip-version-check

pip-licenses --format=markdown

deactivate
rm -rf $VENV/..
