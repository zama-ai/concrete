#!/bin/bash

TOKEN=
ORG_REPO=
EVENTS_TO_CHECK=

while [ -n "$1" ]
do
   case "$1" in
        "--token" )
            shift
            TOKEN="$1"
            ;;

        "--org-repo" )
            shift
            ORG_REPO="$1"
            ;;

        "--event-types" )
            shift
            EVENTS_TO_CHECK="$1"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

# Store the workflows that come in jsons in a file per event type
declare -a JSON_FILES_ARRAY=()
for EVENT in $EVENTS_TO_CHECK; do
  CURR_FILE="$(mktemp --suffix=.json)"
  curl \
  -X GET \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token ${TOKEN}" \
  "https://api.github.com/repos/${ORG_REPO}/actions/runs?branch=main&event=${EVENT}&status=success" | \
  jq -rc '.workflow_runs | sort_by(.updated_at)[-1]' > "${CURR_FILE}"
  JSON_FILES_ARRAY+=("${CURR_FILE}")
done

# Put all the workflows in the same json and dump that
CONCAT_FILE="$(mktemp --suffix=.json)"
jq -sr '.' "${JSON_FILES_ARRAY[@]}" > "${CONCAT_FILE}"

# Sort by updated_at, get the last and get the sha1 for this last one
BEFORE_SHA=$(jq -rc 'sort_by(.updated_at)[-1].head_sha' "${CONCAT_FILE}")

# Remove files
rm "${CONCAT_FILE}"

for FILE_TO_RM in "${JSON_FILES_ARRAY[@]}"; do
  rm "${FILE_TO_RM}"
done

# Echo for the outside world
echo "${BEFORE_SHA}"
