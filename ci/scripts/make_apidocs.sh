#!/usr/bin/env bash

# In this script, we:
#   create a fresh directory
#   create a fresh venv
#   download the last CP
#   make API docs for it

FRESH_DIRECTORY="tempdirectoryforapidocs"

# Keep a copy of the doc, to check changes
rm -rf docs-copy
cp -r docs docs-copy

# Remote old files
rm docs/dev/api/*.md

set -e

# Make a new fresh venv, and install the last public CP there (hence, the --isolated to not take
# nightlies)
rm -rf "$FRESH_DIRECTORY"
mkdir "$FRESH_DIRECTORY"
cd "$FRESH_DIRECTORY"
python3 -m venv .venvtrash
source .venvtrash/bin/activate
pip install concrete-python --index-url https://pypi.org/simple --isolated
pip install lazydocs

# Make API doc files
.venvtrash/bin/lazydocs --output-path="../docs/dev/api" --overview-file="README.md" --src-base-url="../../" --no-watermark concrete
cd -

rm -rf "$FRESH_DIRECTORY"

# New files?
echo "Warning. You might have new API-doc files to git add & push, don't forget"

SED_OPT='-i'
if [ $(uname) == "Darwin" ]; then
    SED_OPT='-i ""'
fi

# Fixing the path issues, to point on files in GitHub
WHICH_PYTHON_VERSION=$(python3 --version | cut -f 2 -d " " | cut -f 1-2 -d ".")
sed $SED_OPT -e "s@../../$FRESH_DIRECTORY/.venvtrash/lib.*/python$WHICH_PYTHON_VERSION/site-packages/@../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/@g" docs/dev/api/concrete.compiler*.md docs/dev/api/concrete.lang*.md
sed $SED_OPT -e "s@../../$FRESH_DIRECTORY/.venvtrash/lib.*/python$WHICH_PYTHON_VERSION/site-packages/@../../frontends/concrete-python/@g" docs/dev/api/concrete.fhe*.md

# Fixing absolute path in doc
sed $SED_OPT -e "s@$PWD/$FRESH_DIRECTORY/@./@g" docs/dev/api/concrete.compiler.library_support.md

# Fixing the links in README.md, which fails (missing .'s for some reason): remove the #headers
sed $SED_OPT -e "s@.md#module-.*)@.md)@g" docs/dev/api/README.md
sed $SED_OPT -e "s@.md#function-.*)@.md)@g" docs/dev/api/README.md
sed $SED_OPT -e "s@.md#class-.*)@.md)@g" docs/dev/api/README.md

# Removed the "object addresses" and "function addresses", since they are not constant
sed $SED_OPT -e "s@object at 0x[a-zA-Z0-9]*@object at ADDRESS@g" docs/*.md
sed $SED_OPT -e "s@object at 0x[a-zA-Z0-9]*@object at ADDRESS@g" docs/*/*.md
sed $SED_OPT -e "s@object at 0x[a-zA-Z0-9]*@object at ADDRESS@g" docs/*/*/*.md

sed $SED_OPT -e "s@function Int at 0x[a-zA-Z0-9]*@function Int at ADDRESS@g" docs/*.md
sed $SED_OPT -e "s@function Int at 0x[a-zA-Z0-9]*@function Int at ADDRESS@g" docs/*/*.md
sed $SED_OPT -e "s@function Int at 0x[a-zA-Z0-9]*@function Int at ADDRESS@g" docs/*/*/*.md

# FIXME: remove this once the PR has been merged once
sed $SED_OPT -e "s@https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt@https://github.com/zama-ai/concrete/blob/main/LICENSE.txt@g" ./docs/dev/api/concrete.lang.dialects.md ./docs/dev/api/concrete.compiler.md ./docs/dev/api/concrete.lang.md

# Was there changes?
if diff -r docs docs-copy; then
    echo ""
else
    echo "There is a difference in the docs, please commit the changes"
    exit -1
fi

# If there were changes, the previous command will stop, thanks to set -e
rm -rf docs-copy

echo "Successful end"

