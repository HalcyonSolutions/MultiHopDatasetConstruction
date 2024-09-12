# Asssumptions:
# Argument 1: path to database dump
DATABASE_DUMP_PATH=$1

# Ensure all parameters are set
if [ -z "$DATABASE_DUMP_PATH" ]; then
    echo "Please provide the path to the database dump as the first argument."
    exit 1
fi

cleanup() {
    echo "Caught Ctrl-C (SIGINT), cleaning up..."
    # Insert your cleanup commands here
    kill -9 $(lsof -t -i:42022)
    exit 0  # Exit cleanly
}

# Will create a new instance based on the container that we have here.
# Ask for name of instance
ZONE="asia-east1-c"
MACHINE_TYPE="e2-standard-4"
CONTAINER_IMAGE="neo4j:5.22.0-community"
USER=$(gcloud config get-value account)
TAGS="benchmarking-env,http-server,https-server"
METADATA="gce-container-image=$CONTAINER_IMAGE"
METADATA="$METADATA,creator=$USER"
DISK_SIZE="200"

echo -e "\033[0;33m"
read -p "Please enter the name of the instance: " INSTANCE_NAME
echo -e "\033[0m"
# Show current constants to user and ask for confirmation.
echo -e "\033[0;33mThe following constants will be used:\033[0m"
echo -e "\033[0;33m- Zone: $ZONE\033[0m"
echo -e "\033[0;33m- Machine Type: $MACHINE_TYPE\033[0m"
echo -e "\033[0;33m- Container Image: $CONTAINER_IMAGE\033[0m"
echo -e "\033[0;33m- Disk Size: $DISK_SIZE\033[0m"
echo -e "\033[0;33m- Instance Name: $INSTANCE_NAME\033[0m"
echo -e "\033[0;33m"
read -p "Are you sure you want to continue? [y/N]: " CONFIRM
echo -e "\033[0m"
if [ "$CONFIRM" != "y" ]; then
    echo "Exiting. If you want to change this variable, please set them as envvars before running this script."
    exit 1
fi

gcloud compute instances create-with-container \
    "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --container-image="$CONTAINER_IMAGE" \
    --tags="$TAGS" \
    --metadata="$METADATA" \
    --boot-disk-size="$DISK_SIZE" \

if [ $? -ne 0 ]; then
  echo -e "\033[0;31m ❌ Instance $INSTANCE_NAME failed to create\033[0m"
else
  echo -e "\033[0;33m ✅ Instance $INSTANCE_NAME creation command successful\033[0m".

  # Now we need to see if we can keep synchronizing the results here.
fi

# Now we wait for the instance to be online
while true; do
  echo -e "\033[0;33m Waiting for instance to be online\033[0m"
  sleep 10
  INSTANCE_STATUS=$(gcloud compute instances describe --zone $ZONE $INSTANCE_NAME --format="get(status)")
  if [ "$INSTANCE_STATUS" == "RUNNING" ]; then
    echo -e "\033[0;33m Instance is online\033[0m"
    break
  fi
done

# Now setup an ssh tunnel
echo -e "\033[0;33m Setting up ssh tunnel\033[0m"
# Use Neo4j ports
gcloud compute ssh --zone $ZONE $INSTANCE_NAME -- -L 7474:localhost:7474 -L 7687:localhost:7687 -f -N
if [ $? -ne 0 ]; then
    echo -e "\033[0;31m ❌ Failed to setup ssh tunnel\033[0m"
    exit 1
fi

# Check if th bucket gs://halcyone-graphs contains freebase/${DATABASE_DUMP_PATH}
# If not, upload the file to the bucket
if ! gsutil ls gs://halcyone-graphs/freebase/${DATABASE_DUMP_PATH}; then
    echo -e "\033[0;33m File not found in bucket, uploading...\033[0m"
    gsutil cp $DATABASE_DUMP_PATH gs://halcyone-graphs/freebase/${DATABASE_DUMP_PATH}
else
    echo -e "\033[0;33m File found in bucket, skipping upload\033[0m"
fi

#TODO: (maybe) check if we have to work with firewall to reach these things

# This is a new instance so we need to import the database
gcloud compute ssh $INSTANCE_NAME -- \
    "sudo docker exec -it neo4j /var/lib/neo4j/bin/neo4j-admin database load --from-path=gs://halcyone-graphs/freebase/${DATABASE_DUMP_PATH}" freebase-wiki --overwrite-destination=true

