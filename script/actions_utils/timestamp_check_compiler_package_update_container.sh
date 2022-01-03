#!/bin/bash -e

set -e

FILE=
COMPILER_RELEASE_ENDPOINT_URL=
ENV_IMG_ENDPOINT_URL=
TOKEN=
ENV_DOCKERFILE=./docker/Dockerfile.concrete-framework-env
GITHUB_ENV_FILE=debug.txt

while [ -n "$1" ]
do
   case "$1" in
        "--file" )
          shift
          FILE="$1"
          ;;

        "--compiler-release-endpoint-url" )
            shift
            COMPILER_RELEASE_ENDPOINT_URL="$1"
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

ENV_JSON=$(curl \
-X GET \
-H "Accept: application/vnd.github.v3+json" \
-H "Authorization: token ${TOKEN}" \
"${ENV_IMG_ENDPOINT_URL}")

LATEST_ENV_IMG_JSON=$(echo "${ENV_JSON}" | \
jq -rc '.[] | select(.metadata.container.tags[] | contains("latest"))')

RELEASE_JSON=$(curl -H "Authorization: token ${TOKEN}" \
-H "Accept: application/vnd.github.v3.raw" \
"${COMPILER_RELEASE_ENDPOINT_URL}" | jq '. | map(select(.draft == false))[0]')

echo "Release json:"
echo "${RELEASE_JSON}"

ASSET_JSON=$(echo "${RELEASE_JSON}" | jq ".assets | map(select(.name | contains(\"${FILE}\")))[0]")

echo "Asset json:"
echo "${ASSET_JSON}"

if [[ "${ASSET_JSON}" == "null" ]]; then
    echo "No asset found matching ${FILE}"
    exit 0
fi

LATEST_ENV_IMG_TIMESTAMP=$(echo "${LATEST_ENV_IMG_JSON}" | jq -r '.updated_at')
LATEST_COMPILER_PACKAGE_TIMESTAMP=$(echo "${ASSET_JSON}" | jq -r '.updated_at')

echo "Latest env image timestamp: ${LATEST_ENV_IMG_TIMESTAMP}"
echo "Latest compiler package timestamp: ${LATEST_COMPILER_PACKAGE_TIMESTAMP}"

LATEST_ENV_IMG_EPOCH=$(date -d "${LATEST_ENV_IMG_TIMESTAMP}" +%s)
LATEST_COMPILER_PACKAGE_EPOCH=$(date -d "${LATEST_COMPILER_PACKAGE_TIMESTAMP}" +%s)

echo "Latest env image epoch: ${LATEST_ENV_IMG_EPOCH}"
echo "Latest compiler package epoch: ${LATEST_COMPILER_PACKAGE_EPOCH}"

if [[ "${LATEST_COMPILER_PACKAGE_EPOCH}" -gt "${LATEST_ENV_IMG_EPOCH}" ]]; then
    echo "Env image out of date, sending rebuild request."
    TMP_DOCKER_FILE="$(mktemp)"
    sed "s/\(# compiler timestamp: \)\(.*\)/\1${LATEST_COMPILER_PACKAGE_TIMESTAMP}/g" \
        "${ENV_DOCKERFILE}" > "${TMP_DOCKER_FILE}"
    cp -f "${TMP_DOCKER_FILE}" "${ENV_DOCKERFILE}"
    rm -f "${TMP_DOCKER_FILE}"
    echo "LATEST_COMPILER_PACKAGE_TIMESTAMP=${LATEST_COMPILER_PACKAGE_TIMESTAMP}" \
    >> "${GITHUB_ENV_FILE}"
    echo "New package timestamp: ${LATEST_COMPILER_PACKAGE_TIMESTAMP}"
else
    echo "Image up to date, nothing to do."
fi
