#!/usr/bin/env bash

set -e

# Read the hash from stdin
hash=""
while IFS= read -r line; do
    hash="$line"
done

is_sha256() {
    local input="$1"
    if [[ "$input" =~ ^[A-Fa-f0-9]{64}$ ]]; then
      echo "$input"
    else
      echo ""
    fi
}
# Function to read the first 256 bits (64 hex characters) from a file
#TODO: This is but a proxy. I dont like it but it signifies leess work for now.
read_first_256_bits() {
    local file_path="$1"
    if [[ -f "$file_path" ]]; then
        # Read the first 64 characters from the file
        local first_256_bits
        first_256_bits=$(head -c 64 "$file_path" | xxd -p | tr -d '\n')
        echo "$first_256_bits"
    else
        echo "File not found"
        exit 1
    fi
}


# Will provide $DATA_BUCKET_NAME and 
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
source "$REPO_ROOT_DIR/.env"

# Define the file path
file_path="$1"

# Check if the file already exists
if [ -f "$file_path" ]; then
    # If the file exists, verify its integrity by checking the hash
    existing_hash=$(sha256sum "$file_path" | awk '{print $1}')
    is_hash=$(is_sha256 "$(read_first_256_bits "$file_path")")
    current_commit=$(git log -n 1 --pretty=format:"%H" -- "$file_path")
    if [ -z "$is_hash" ]; then
        echo -e "\033[033m Failure: $file_path is not a hash, yet it is part of .gitattributes. Please contact author of commit ${current_commit}"
    elif [ "$existing_hash" == "$hash" ]; then
        echo "File already exists and matches hash. Skipping download."
        exit 0
    else
        echo "File ${file_path} exists but does not match hash. Downloading fresh copy." >&2
    fi
fi

# Ensure the cache directory exists
mkdir -p "$LFS_CACHE_DIRECTORY"

# Download the file from GCS if not already cached
if [ ! -f "$LFS_CACHE_DIRECTORY/$hash" ]; then
    gsutil cp "$DATA_BUCKET_NAME/${file_path}.${hash}" "$LFS_CACHE_DIRECTORY/$hash"
fi

# Copy the downloaded file from cache to the correct file path
cat "$LFS_CACHE_DIRECTORY/$hash"
