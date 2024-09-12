# List all instances started by current user
CURRENT_USER=$(gcloud config get-value account)
echo "Listing all instances started by $CURRENT_USER"
gcloud compute instances list --filter="serviceAccounts[0].email=$CURRENT_USER"

