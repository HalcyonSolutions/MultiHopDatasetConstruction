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
# Start an ssh-agent to avoid PITA later
eval $(ssh-agent -s)

# Will create a new instance based on the container that we have here.
# Ask for name of instance
ZONE="asia-east1-c"
MACHINE_TYPE="e2-standard-4"
CONTAINER_IMAGE="neo4j:5.22.0-community"
GUSER=$(gcloud config get-value account)
TAGS="benchmarking-env,http-server,https-server"
METADATA="gce-container-image=$CONTAINER_IMAGE"
METADATA="$METADATA,creator=$GUSER"
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
    --container-mount-host-path=host-path=/tmp/data_dump/,mount-path=/home/neo4j,mode=rw 


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
  echo -n "."
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
# Bind cleanup
echo "Tunnel setup"
trap cleanup SIGINT
if [ $? -ne 0 ]; then
    echo -e "\033[0;31m ❌ Failed to setup ssh tunnel\033[0m"
    exit 1
fi

DATABASE_DUMP_NAME=$(basename $DATABASE_DUMP_PATH)
# Check if th bucket gs://halcyone-graphs contains freebase/${DATABASE_DUMP_PATH}
# If not, upload the file to the bucket
echo -e "\033[0;33m Checking if file is in bucket\033[0m"
if ! gsutil ls gs://halcyon-graphs/freebase/${DATABASE_DUMP_NAME}; then
    #TODO: For now this is only for backup. We would likely want to use this for the actual import
    echo -e "\033[0;33m File not found in bucket, uploading...\033[0m"
    gsutil cp $DATABASE_DUMP_PATH gs://halcyon-graphs/freebase/${DATABASE_DUMP_NAME}
else
    echo -e "\033[0;33m::File found in bucket, skipping upload\033[0m"
fi

#TODO: (maybe) check if we have to work with firewall to reach these things
echo "Getting some metadata"
PUBLIC_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
# TODO: Hacky, potentially dangerous. I dont like it but I am at my wits end:
echo -e "\033[0;33m::Setting up permissions for user $USER \033[0m"
gcloud compute ssh --zone "$ZONE" "$INSTANCE_NAME" --command="sudo usermod -aG root $USER && sudo chmod 771 /tmp/data_dump"

echo -e "\033[0;33m::Copying file to instance\033[0m"
gcloud compute scp $DATABASE_DUMP_PATH "${INSTANCE_NAME}:/tmp/data_dump/${DATABASE_DUMP_NAME}" --zone "$ZONE"

# Ask docker for the container id of the neo4j container
CONTAINER_ID=$(gcloud compute ssh --zone "$ZONE" "$INSTANCE_NAME" --command="sudo docker ps -q -l")
echo -e "\033[0;33m::Container ID is $CONTAINER_ID\033[0m"

# This is a new instance so we need to import the database
echo -e "\033[0;33m::Importing database\033[0m"
# Remove this .dump extension from dump_name
basename_dump_name=$(basename $DATABASE_DUMP_NAME .dump)
gcloud compute ssh --zone "$ZONE" "$INSTANCE_NAME" -- \
    "sudo docker exec -it ${CONTAINER_ID} /var/lib/neo4j/bin/neo4j-admin database load --from-path=/home/neo4j/ ${DATABASE_DUMP_NAME%.dump}" --overwrite-destination=true
echo -e "\033[0;33m::Database imported\033[0m"
echo -e "\033[0;33m::Script Ended\033[0m"
