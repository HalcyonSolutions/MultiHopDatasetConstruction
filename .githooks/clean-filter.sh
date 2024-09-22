#!/usr/bin/env bash

set -e

echo "Cleaning up file: $1" >> loggity.log

# Read the file path from the argument
file_path="$1"
repo_root_dir=$(git rev-parse --show-toplevel)
# Will provide $DATA_BUCKET_NAME and 
source "$repo_root_dir/.env"


# Calculate the SHA256 hash of the file
hash=$(sha256sum "$file_path" | awk '{print $1}')

# Upload the file to GCS using the hash as the key
gsutil_command="gsutil cp ${file_path} ${DATA_BUCKET_NAME}/${file_path}.${hash}"
echo "Running gsutil command: $gsutil_command"  >> loggity.log
$gsutil_command

# Output the hash to Git as the placeholder content
echo "$hash"
