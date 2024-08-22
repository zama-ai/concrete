#!/usr/bin/env bash

# We don't want links to the main branch, even if it's public. Instead, release branch should be
# targeted
if grep -r "tree/main" ../../docs | grep "\.md:" | grep -v "https://huggingface.co/spaces/"; then
    echo -n -e "\nThe above links contain references to the main banch. Please remove them as only "
    echo "release branches should be referenced."
    exit 255
fi


# We don't want links to our internal repositories (Concrete ML or Concrete), expect if they are
# GitHub issues
if grep -r "concrete-ml-internal" ../../docs | grep "\.md:" | grep -v "concrete-ml-internal/issues"; then
    echo -n -e "\nThe above links contain references to the 'concrete-ml-internal' private "
    echo -n -e "repository that are not issues. Please remove them as only the 'concrete-ml' "
    echo "public should be referenced."
    exit 255
fi

if grep -r "concrete-internal" ../../docs | grep "\.md:"; then
    echo -n -e "\nThe above links contain references to the 'concrete-internal' private "
    echo -n -e "repository that are not issues. Please remove them as only the 'concrete' "
    echo "public should be referenced."
    exit 255
fi

if grep -r app.gitbook.com ../../docs; then
    echo -e "\nThe above links contain references to our (internal) GitBook. Please remove them."
    exit 255
fi
