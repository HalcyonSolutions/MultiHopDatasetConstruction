# When  you launch this it will keep an eye on your files and synchronize them to your container whenever a change occurs.

cleanup() {
    echo "Caught Ctrl-C (SIGINT), cleaning up..."
    # Insert your cleanup commands here
    kill -9 $(lsof -t -i:42022)
    exit 0  # Exit cleanly
}

# Ensure there is an ssh agent running
echo "Starting ssh agent. We will likely need to enter your password here."
eval $(ssh-agent -s)
ssh-add ./collective_key

# We assume:
DESIRED_INSTANCE=$1
INSTANCE_ZONE=$2
FOLDER_TO_SYNC=$3

# We will now get the public ip.
PUBLIC_IP=$(gcloud compute instances describe "$DESIRED_INSTANCE" --zone="$INSTANCE_ZONE" --format=json | jq -r '.networkInterfaces[0].accessConfigs[0].natIP')

# We will ensure that we have access to the remote with a specific key.
gcloud compute ssh "$DESIRED_INSTANCE" --zone="$INSTANCE_ZONE" --command="docker logs \$(docker ps -l -q)" -- -L 8888:localhost:8888 -L 42022:localhost:42022 -f -N
echo -e "\033[0;33m::Set ssh tunnel to ${DESIRED_INSTANCE}::${PUBLIC_IP}:8888\033[0m"

# Now rsync will be running in the background between these two directories. 
REMOTE_ROOT_DIR="/opt/nlpbench"
GET_GIT_ROOT_DIR=$(git rev-parse --show-toplevel)

echo "Starting to watch files in $GET_GIT_ROOT_DIR"

trap cleanup SIGINT
while true; do
  # We will now run the rsync command
  echo -e "\033[0;33m::Syncing files to ${DESIRED_INSTANCE}\033[0m"
  rsync -e 'ssh -i ./collective_key -p 42022' -avz --exclude-from="${GET_GIT_ROOT_DIR}/.rsync_exclude" "${GET_GIT_ROOT_DIR}/" itl-operator@localhost:${REMOTE_ROOT_DIR} 
  sleep 5
done

# Kill the ssh tunnel
kill -9 $(lsof -t -i:42022)
