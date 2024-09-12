#!/bin/bash
# This script will synchronize to the given instance or latest instance.
# It assumes that the instance is running and that the user has access to it.

echo "Start" > log.log
CURRENT_USER=$(gcloud config get-value account)
INSTANCE_INFO="[]"
if [ -z "$1" ]; then
  echo -e "\033[0;33m::No instance name provided. Will use the latest instance\033[0m"
  INSTANCES=$(gcloud compute instances list --filter="status=RUNNING" --format=json)
  SORTD_INSTANCES=$(echo "$INSTANCES" | jq --arg CURRENT_USER "$CURRENT_USER" -r '.[] | select(.metadata.items[].value == $CURRENT_USER) | {name, lastStartTimestamp, zone: .zone}' | jq  -r -s 'sort_by(.lastStartTimestamp) | reverse ')
  if [ "$INSTANCES" == "[]" ]; then
    echo -e "\033[0;31m::No instance found. Exiting\033[0m"
    exit 1
  fi
  INSTANCE_INFO=$(echo "$SORTD_INSTANCES" | jq -r '.[0]')
else
  INSTANCE_INFO=$(gcloud compute instances list --filter="name=${instance_name} and status=RUNNING" --format=json |  jq -r '.[] | {name, lastStartTimestamp, zone: .zone}')
  # Assert instance info is not empty
  if [ -z "$INSTANCE_INFO" ]; then
    echo -e "\033[0;31m::No instance found. Exiting\033[0m"
    exit 
  fi
  # Corroborate that the instance is available to the user 
fi

INSTANCE_NAME=$(echo "$INSTANCE_INFO" | jq -r '.name')
INSTANCE_ZONE=$(echo "$INSTANCE_INFO" | jq -r '.zone')
echo -e "\033[0;33m Syncing files to instance '${INSTANCE_NAME}'\033[0m"

## Important Call Here
echo -e "\033[0;33m Checking for instance to be up and running\033[0m"
time_out=30 # in seconds
waiting_time=0
while true; do
  # Check if instance is up and running
  INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$INSTANCE_ZONE" --format=json)
  INSTANCE_STATUS=$(echo "$INSTANCE_STATUS" | jq -r '.status')
  if [ "$INSTANCE_STATUS" == "RUNNING" ]; then
    break
  fi
  sleep 5
  waiting_time=$((waiting_time+5))
  if [ $waiting_time -ge $time_out ]; then
    echo -e "\033[0;31mInstance $instance_name failed to start\033[0m"
    exit 1
  fi
done

echo -e "\033[0;33m Syncing files to istance\033[0m"

rsync -avz --exclude-from="${REPO_ROOT_DIR}/${FILEW_EXCLUDE_PATTERNS}" "${REPO_ROOT_DIR}/" "${instance_name}:/opt/nlpbench"

