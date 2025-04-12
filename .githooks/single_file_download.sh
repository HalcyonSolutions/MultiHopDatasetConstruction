#!/usr/bin/env bash
# Usage: single_file_download.sh <file_to_download> [commit_hash]
# When receiving a single file it will try to download it in place.
# If a commit hash is provided, it will use the file content at that commit.
set -e

# Read the file to download and optional commit hash
FILE_TO_DOWNLOAD=$1
COMMIT_HASH=$2

# Determine the hash inside the file
if [ -z "$COMMIT_HASH" ]; then
    # No commit hash provided, use the current file content
    HASH_INSIDE_FILE=$(cat "$FILE_TO_DOWNLOAD")
else
    # Commit hash provided, get the file content at that commit
    HASH_INSIDE_FILE=$(git show "$COMMIT_HASH:$FILE_TO_DOWNLOAD")
fi

echo "Reading Cache inside of file with hash: $HASH_INSIDE_FILE"

# Will provide $DATA_BUCKET_NAME and 
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
source "$REPO_ROOT_DIR/.env"

# Check if the file already exists
if [ ! -z "$HASH_INSIDE_FILE" ]; then
    # If the file exists, verify its integrity by checking the hash
    # Ensure the cache directory exists
    mkdir -p "$LFS_CACHE_DIRECTORY"

    # Download the file from GCS if not already cached
    if [ ! -f "$LFS_CACHE_DIRECTORY/$HASH_INSIDE_FILE" ]; then
        gsutil cp "$DATA_BUCKET_NAME/${FILE_TO_DOWNLOAD}.${HASH_INSIDE_FILE}" "$LFS_CACHE_DIRECTORY/$HASH_INSIDE_FILE"
        echo "Downloaded file to $LFS_CACHE_DIRECTORY/$HASH_INSIDE_FILE"
    else
        echo "File already exists in cache. Copying from cache."
    fi

    # Then Back to its original position 
    cp "$LFS_CACHE_DIRECTORY/$HASH_INSIDE_FILE" "$FILE_TO_DOWNLOAD"
    echo "Copied file to $FILE_TO_DOWNLOAD"
fi
