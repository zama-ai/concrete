#!/bin/bash -e

set -e

BASE_IMG_ENDPOINT_URL=
ENV_IMG_ENDPOINT_URL=
TOKEN=
ORG_REPO=
EVENT_TYPE=

while [ -n "$1" ]
do
   case "$1" in
        "--base_img_url" )
            shift
            BASE_IMG_ENDPOINT_URL="$1"
            ;;

        "--env_img_url" )
            shift
            ENV_IMG_ENDPOINT_URL="$1"
            ;;

        "--token" )
            shift
            TOKEN="$1"
            ;;

        "--org-repo" )
            shift
            ORG_REPO="$1"
            ;;

        "--event-type" )
            shift
            EVENT_TYPE="$1"
            ;;

        *)
            echo "Unknown param : $1"
            exit -1
            ;;
   esac
   shift
done

BASE_JSON=$(curl \
-X GET \
-H "Accept: application/vnd.github.v3+json" \
-H "Authorization: token ${TOKEN}" \
"${BASE_IMG_ENDPOINT_URL}")

BASE_IMG_TIMESTAMP=$(echo "${BASE_JSON}" | jq -r 'sort_by(.updated_at)[-1].updated_at')

ENV_JSON=$(curl \
-X GET \
-H "Accept: application/vnd.github.v3+json" \
-H "Authorization: token ${TOKEN}" \
"${ENV_IMG_ENDPOINT_URL}")

ENV_IMG_TIMESTAMP=$(echo "${ENV_JSON}" | \
jq -rc '.[] | select(.metadata.container.tags[] | contains("latest")).updated_at')

echo "Base timestamp: ${BASE_IMG_TIMESTAMP}"
echo "Env timestamp:  ${ENV_IMG_TIMESTAMP}"

BASE_IMG_DATE=$(date -d ${BASE_IMG_TIMESTAMP} +%s)
ENV_IMG_DATE=$(date -d ${ENV_IMG_TIMESTAMP} +%s)

echo "Base epoch: ${BASE_IMG_DATE}"
echo "Env epoch:  ${ENV_IMG_DATE}"

if [[ "${BASE_IMG_DATE}" -ge "${ENV_IMG_DATE}" ]]; then
    echo "Env image out of date, sending rebuild request."
    curl \
    -X POST \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token ${TOKEN}" \
    https://api.github.com/repos/${ORG_REPO}/dispatches \
    -d "{\"event_type\":\"${EVENT_TYPE}\"}"
else
    echo "Image up to date, nothing to do."
fi
