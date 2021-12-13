#!/usr/bin/env bash
#
# Adapted from https://gist.github.com/maxim/6e15aa45ba010ab030c4
#
# gh-dl-release! It works!
#
# This script downloads an asset from latest or specific Github release of a
# private repo. Feel free to extract more of the variables into command line
# parameters.
#
# PREREQUISITES
#
# curl, wget, jq
#
# USAGE
#
# Set all the variables inside the script, make sure you chmod +x it, then
# to download specific version to my_app.tar.gz:
#
#     gh-dl-release 2.1.1 my_app.tar.gz
#
# to download latest version:
#
#     gh-dl-release latest latest.tar.gz
#
# If your version/tag doesn't match, the script will exit with error.

TOKEN=
ORG_REPO=
# the name of your release asset file, e.g. build.tar.gz
FILE=
DEST_DIR=
VERSION="latest"
COMPILER_TAG_OUTPUT_FILE=debug.txt
GITHUB_ENV_FILE=debug.txt

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

        "--version" )
          shift
          VERSION="$1"
          ;;

        "--dest-dir" )
          shift
          DEST_DIR="$1"
          ;;

        "--github-env")
            shift
            GITHUB_ENV_FILE="$1"
            ;;

        "--file" )
          shift
          FILE="$1"
          ;;

        "--compiler-tag-output-file")
            shift
            COMPILER_TAG_OUTPUT_FILE="$1"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

alias errcho='>&2 echo'

mkdir -p "${DEST_DIR}"

if [[ "${VERSION}" == "latest" ]]; then
  # Select first non draft version
  jq_parser='. | map(select(.draft == false))[0]'
else
  jq_parser=". | map(select(.tag_name == \"${VERSION}\"))[0]"
fi;

release_json=$(curl -H "Authorization: token ${TOKEN}" \
-H "Accept: application/vnd.github.v3.raw" \
"https://api.github.com/repos/${ORG_REPO}/releases" | jq "${jq_parser}")

echo "Release json:"
echo "${release_json}"

asset_json=$(echo "${release_json}" | jq ".assets | map(select(.name | contains(\"${FILE}\")))[0]")

echo "Asset json:"
echo "${asset_json}"

asset_filename=$(echo "${asset_json}" | jq -rc '.name')
echo "Asset filename:"
echo "${asset_filename}"
echo "WHEEL=${asset_filename}" >> "${GITHUB_ENV_FILE}"

release_tag=$(echo "${release_json}" | jq -rc '.tag_name')
asset_id=$(echo "${asset_json}" | jq -rc '.id')

release_tag="${release_tag//-/_}"

echo "Release tag: ${release_tag}"
echo "Asset id: ${asset_id}"

if [[ "${asset_id}" == "null" ]]; then
  errcho "ERROR: version not found ${VERSION}"
  exit 1
fi

echo "Downloading..."

wget --auth-no-challenge --header='Accept:application/octet-stream' \
  "https://${TOKEN}:@api.github.com/repos/${ORG_REPO}/releases/assets/${asset_id}" \
  -O "${DEST_DIR}/${asset_filename}"

err_code=$?

echo "Done."
echo "${release_tag}" >> "${COMPILER_TAG_OUTPUT_FILE}"

exit "${err_code}"
