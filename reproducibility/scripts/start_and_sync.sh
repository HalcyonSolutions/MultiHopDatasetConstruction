#!/bin/bash


_get_metadata_elem(){
  ROWS="$1"
  TARGET_KEY="$2"
  return_value=""
  # echo "ROWS: ${ROWS}" >> log.log
  IFS=$'\n' 
  while read -r elem; do
    CUR_KEY=$(echo ${elem} | base64 --decode | jq -r '.key')
    CUR_VALUE=$(echo ${elem} | base64 --decode | jq -r '.value')
    # echo "TARGET (${TARGET_KEY}): eleme: ${elem}" >> log.log
    # echo "TARGET (${TARGET_KEY}): CUR_KEY: ${CUR_KEY}" >> log.log
    # echo "TARGET (${TARGET_KEY}): CUR_VALUE: ${CUR_VALUE}" >> log.log
    if [ "$CUR_KEY" == "$TARGET_KEY" ]; then
      return_value="$CUR_VALUE"
      break
    fi
  done <<< "$ROWS"
  echo "$return_value" #CHECK:
}

# USER Constants
CONTAIMER_TAG="asia-east1-docker.pk.dev/optical-loop-431606-m6/kb-reistry/barebones1 "

# Constants
LOCAL_ROOT_DIR=$(pwd)
REMOTE_COMMAND=$0
FILEW_EXCLUDE_PATTERNS="./.rsync_exclude" # Assuming run from root of repo
# Set Repo Root dir to be the parent of the first .git/ directory in the path
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)

# Remote data
CURRENT_USER=$(gcloud config get-value account)
# Get non-running instances
echo ":: Looking for instances for user $CURRENT_USER"
# TOREM: b4 production
AVAILABLE_INSTANCES=$(gcloud compute instances list --filter="status=TERMINATED" --format=json)
INSTANCE_NAMES_AND_METADATA=$(echo "$AVAILABLE_INSTANCES" | jq -r '.[] | select(.metadata.items[].key == "gce-container-declaration") | {name, metadata: .metadata, zone: .zone} | @base64')
echo "$INSTANCE_NAMES_AND_METADATA"  | base64 --decode > instances.json 

# This will be our options
declare -a row_strings

# CHECK: Why the need of base64 encoding
echo "Processing ${INSTANCE_NAMES_AND_METADATA[@]}" >> log.log
counter=0
for row in $INSTANCE_NAMES_AND_METADATA; do
    decoded_row=$(echo "${row}" | base64 --decode)

    metadata_items=$(jq -r '.metadata.items[] | @base64' <<< "$decoded_row")
    CREATOR=$(_get_metadata_elem "$metadata_items" "creator")

    if [ "$CURRENT_USER" != "$CREATOR" ]; then
      continue
    fi
    # Otherwise add the info to row_strings
    INSTANCE_NAME=$(echo "${row}" | base64 --decode | jq -r ".name")

    # NOTE: This is not official API So it may break
    gce_container_declaration=$(_get_metadata_elem "$metadata_items" "gce-container-declaration")

    # NOTE: Here we make the assumption that there is only one container in the yaml
    IMAGE_NAME=$(echo "${gce_container_declaration}" | yq -r '.spec.containers[0].image')
    clean_name=$(echo "$IMAGE_NAME" | awk -F"/" '{print $NF}')

    ZONE_NAME=$( jq -r ".zone" <<< "$decoded_row" | awk -F"/" '{print $NF}')

    row_strings+=("${counter}\t${INSTANCE_NAME}\t${ZONE_NAME}\t${IMAGE_NAME}\n")
    counter=$((counter+1))

done

echo -e "\033[0;33m::The following instances are available to you: \033[0m"
# Table Clumn titles
COLUMN_TITLES="\033[0;33mID\tInstance Name\t Zone\tImage Name\033[0m"
OUTPUT=$(echo -e "$COLUMN_TITLES\n${row_strings[@]}")
column -t -s $'\t'  <<< "$OUTPUT"

# Get User Inputs and resulting info
read -p "Enter the number of the instance you want to use: " INSTANCE_NUM
instance_name=$(echo -e "${row_strings[$INSTANCE_NUM]}" | cut -d$'\t' -f2)
instance_zone=$(echo -e "${row_strings[$INSTANCE_NUM]}" | cut -d$'\t' -f3)

## Important Call Here
gcloud_command="gcloud compute instances start ${instance_name} --zone=${instance_zone}"
echo -e "\033[0;33m Running '${gcloud_command}'\033[0m"
$gcloud_command
# Check on result
if [ $? -neq 0 ]; then
  echo -e "\033[0;31m❌Instance '${instance_name}' failed to start\033[0m"
fi

# TODO: Add start script
