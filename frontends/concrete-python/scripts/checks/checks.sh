#!/bin/bash

set -ex

cd frontends/concrete-python
make venv
source .venv/bin/activate
make pcc
