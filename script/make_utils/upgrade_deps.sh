#!/usr/bin/env bash

# verbose output please
set -v

no_dev_file=$(mktemp --suffix=.txt)
all_file=$(mktemp --suffix=.txt)
dev_file=$(mktemp --suffix=.txt)

poetry show -o -t --no-dev | grep -v -e "--" | cut -d " " -f 1 | sed 's/$/\@latest/g' > "${no_dev_file}"
poetry show -o -t | grep -v -e "--" | cut -d " " -f 1 | sed 's/$/\@latest/g' > "${all_file}"
join -v1 -v2 "${all_file}" "${no_dev_file}" > "${dev_file}"
# shellcheck disable=SC2002
cat "${no_dev_file}" | xargs poetry add
# shellcheck disable=SC2002
cat "${dev_file}" | xargs poetry add --dev

rm "${no_dev_file}"
rm "${dev_file}"
rm "${all_file}"
