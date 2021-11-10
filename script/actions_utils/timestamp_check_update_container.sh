#!/bin/bash -e

set -e

BASE_IMG_ENDPOINT_URL=
ENV_IMG_ENDPOINT_URL=
TOKEN=
ENV_DOCKERFILE=./docker/Dockerfile.concretefhe-env
GITHUB_ENV_FILE=debug.txt

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

        "--github-env")
            shift
            GITHUB_ENV_FILE="$1"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

BASE_JSON=$(curl \
-X GET \
-H "Accept: application/vnd.github.v3+json" \
-H "Authorization: token ${TOKEN}" \
"${BASE_IMG_ENDPOINT_URL}")

LATEST_BASE_IMG_JSON=$(echo "${BASE_JSON}" | jq -rc 'sort_by(.updated_at)[-1]')

echo "Latest base image json: ${LATEST_BASE_IMG_JSON}"

LATEST_BASE_IMG_TIMESTAMP=$(echo "${LATEST_BASE_IMG_JSON}" | jq -r '.updated_at')

ENV_JSON=$(curl \
-X GET \
-H "Accept: application/vnd.github.v3+json" \
-H "Authorization: token ${TOKEN}" \
"${ENV_IMG_ENDPOINT_URL}")

LATEST_ENV_IMG_JSON=$(echo "${ENV_JSON}" | \
jq -rc '.[] | select(.metadata.container.tags[] | contains("latest"))')

ENV_IMG_TAG=$(echo "${LATEST_ENV_IMG_JSON}" | \
jq -rc '.metadata.container.tags - ["latest"] | .[0]')

echo "env image tag: ${ENV_IMG_TAG}"

IFS='-' read -ra ENV_IMG_SPLIT <<< "${ENV_IMG_TAG}"
ENV_IMG_BASE_IMG_TAG="${ENV_IMG_SPLIT[0]}"

echo "env image base image tag: ${ENV_IMG_BASE_IMG_TAG}"

CURRENT_BASE_IMG_JSON=$(echo "${BASE_JSON}" | \
jq -rc ".[] | select(.metadata.container.tags[] | contains(\"${ENV_IMG_BASE_IMG_TAG}\"))")

echo "current base image json: ${CURRENT_BASE_IMG_JSON}"

CURRENT_BASE_IMG_TIMESTAMP=$(echo "${CURRENT_BASE_IMG_JSON}" | jq -r '.updated_at')

echo "Latest base timestamp: ${LATEST_BASE_IMG_TIMESTAMP}"
echo "Current base timestamp:  ${CURRENT_BASE_IMG_TIMESTAMP}"

LATEST_BASE_IMG_TIMESTAMP=$(date -d "${LATEST_BASE_IMG_TIMESTAMP}" +%s)
CURRENT_BASE_IMG_DATE=$(date -d "${CURRENT_BASE_IMG_TIMESTAMP}" +%s)

echo "Base epoch: ${LATEST_BASE_IMG_TIMESTAMP}"
echo "Env epoch:  ${CURRENT_BASE_IMG_DATE}"

if [[ "${LATEST_BASE_IMG_TIMESTAMP}" -gt "${CURRENT_BASE_IMG_DATE}" ]]; then
    echo "Env image out of date, sending rebuild request."
    NEW_BASE_IMG_TAG=$(echo "${LATEST_BASE_IMG_JSON}" | \
    jq -rc '.metadata.container.tags - ["latest"] | .[0]')
    echo "NEW_BASE_IMG_TAG=${NEW_BASE_IMG_TAG}" >> "${GITHUB_ENV_FILE}"
    echo "New base img tag: ${NEW_BASE_IMG_TAG}"
    TMP_DOCKER_FILE="$(mktemp)"
    sed "s/\(FROM\ ghcr\.io\/zama-ai\/zamalang-compiler:\)\(.*\)/\1${NEW_BASE_IMG_TAG}/g" \
        "${ENV_DOCKERFILE}" > "${TMP_DOCKER_FILE}"
    cp -f "${TMP_DOCKER_FILE}" "${ENV_DOCKERFILE}"
    rm -f "${TMP_DOCKER_FILE}"
else
    echo "Image up to date, nothing to do."
fi
