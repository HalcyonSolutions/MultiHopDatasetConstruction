INSTANCE_NAME=$1

# Get the IP of the instance
INSTANCE_IP=$(gcloud compute instances describe $INSTANCE_NAME --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
CURRENT_USER=$(gcloud config get-value account)
TARGET_DIR="/opt/nlpbench"

# We have to ensure tha the 

# Get the user's ssh key
SSH_KEY=$(cat $HOME_DIR/.ssh/id_rsa.pub)

# Get the user's ssh key
SSH_KEY=$(cat $HOME_DIR/.ssh/id_rsa.pub)

echo "ssh -i $HOME_DIR/.ssh/id_rsa -o StrictHostKeyChecking=no $CURRENT_USER@$INSTANCE_IP"

ssh -i $HOME_DIR/.ssh/id_rsa -o StrictHostKeyChecking=no $CURRENT_USER@$INSTANCE_IP
