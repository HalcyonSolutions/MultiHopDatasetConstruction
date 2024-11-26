#!/usr/bin/env bash
# Usage: single_file_download.sh <file_to_download>
# When receiving a single file it wilil try to dload it in place. 
set -e
#
# Read the hash from stdin
FILE_TO_DOWNLOAD=$1
HASH_INSIDE_FILE=$(cat $FILE_TO_DOWNLOAD)

echo 'Reading Cache inside of file'

# Will provide $DATA_BUCKET_NAME and 
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
source "$REPO_ROOT_DIR/.env"

# Check if the file already exists
if [ -f "$FILE_TO_DOWNLOAD" ]; then
    # If the file exists, verify its integrity by checking the hash
    CACHE_DIR=".gcs_cache"
    #
    # Ensure the cache directory exists
    mkdir -p "$CACHE_DIR"

    # Download the file from GCS if not already cached
    if [ ! -f "$CACHE_DIR/$HASH_INSIDE_FILE" ]; then
        gsutil cp "$DATA_BUCKET_NAME/${FILE_TO_DOWNLOAD}.${HASH_INSIDE_FILE}" "$CACHE_DIR/$HASH_INSIDE_FILE"
    fi

    # Then Back to its original position 
    cp "$CACHE_DIR/$HASH_INSIDE_FILE" "$FILE_TO_DOWNLOAD"
fi

