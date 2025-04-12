#!/usr/bin/env bash

set -e

# Get some important shared variables
#  Like:
#  - LFS_CACHE_DIRECTORY
GIT_REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_REPO_ENV="${GIT_REPO_ROOT}/.env"
source "$GIT_REPO_ENV"
mkdir -p "${LFS_CACHE_DIRECTORY}"


FILE_PATH="$1" # Read the file path from the argument
HASH=$(sha256sum "$FILE_PATH" | awk '{print $1}')
CACHE_PATH="${LFS_CACHE_DIRECTORY}/${HASH}"

# Check if the file already exists in the cache
if [[ ! -f "$CACHE_PATH" ]]; then 
  cp "$FILE_PATH" "$CACHE_PATH"
fi

# Upload the file to GCS using the hash as the key
# gsutil_command="gsutil cp ${file_path} ${DATA_BUCKET_NAME}/${file_path}.${HASH}"
# $gsutil_command

# Output the hash to Git as the placeholder content
echo "$HASH"
exit 0
