#!/usr/bin/env bash

set -e

# Read the hash from stdin
hash=""
while IFS= read -r line; do
    hash="$line"
done


# Will provide $DATA_BUCKET_NAME and 
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
source "$REPO_ROOT_DIR/.env"

# Define the file path
file_path="$1"

# Check if the file already exists
if [ -f "$file_path" ]; then
    # If the file exists, verify its integrity by checking the hash
    existing_hash=$(sha256sum "$file_path" | awk '{print $1}')
    if [ "$existing_hash" == "$hash" ]; then
        echo "File already exists and matches hash. Skipping download."
        exit 0
    else
        echo "File exists but does not match hash. Downloading fresh copy."
    fi
fi

# Define a cache directory to store downloaded files (optional)
CACHE_DIR=".gcs_cache"

# Ensure the cache directory exists
mkdir -p "$CACHE_DIR"

# Download the file from GCS if not already cached
if [ ! -f "$CACHE_DIR/$hash" ]; then
    gsutil cp "$DATA_BUCKET_NAME/${file_path}.${hash}" "$CACHE_DIR/$hash"
fi

# Copy the downloaded file from cache to the correct file path
cat "$CACHE_DIR/$hash"
