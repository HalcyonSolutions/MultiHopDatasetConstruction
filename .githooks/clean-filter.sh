#!/usr/bin/env bash

set -e

echo "Cleaning up file: $1" >> loggity.log

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

# Read the file path from the argument
file_path="$1"
repo_root_dir=$(git rev-parse --show-toplevel)
# Will provide $DATA_BUCKET_NAME and 
source "$repo_root_dir/.env"

# Check if file only contains 65 characters
num_characters=$(wc -c "$file_path" | awk '{print $1}')
if [[ "$num_characters" -eq 65 ]]; then
  first_256_bits=$(read_first_256_bits "$file_path")
  if [[ -n "$first_256_bits" ]]; then
      echo "$first_256_bits"
      exit
  fi
fi

# Calculate the SHA256 hash of the file
hash=$(sha256sum "$file_path" | awk '{print $1}')

# Upload the file to GCS using the hash as the key
gsutil_command="gsutil cp ${file_path} ${DATA_BUCKET_NAME}/${file_path}.${hash}"
echo "Running gsutil command: $gsutil_command"  >> loggity.log
$gsutil_command

# Output the hash to Git as the placeholder content
echo "$hash"
