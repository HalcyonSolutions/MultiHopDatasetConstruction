#!/bin/bash
# The VM will be in google cloud, so assume gcloud is installed
# This file will rsync a specified folder to the root of the VM.
# Then it will run the command line specified

# The first argument is the folder to be rsynced
# The second argument is the command line to be run
# Third argument will be a comma separated list of folders to be excluded

FOLDER="$1" # e.g. triplet_creations
COMMAND="$2" # e.g. python graph_builder_rdf.py
EXTRA_EXCLUDES="$3" # e.g. data,logs,results,tmp,.git,results,checkpoints,tmp

CONFIG_FILE="gcloud_config.env"

## Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "yq is not installed. Please install it before continuing."
    exit 1
fi
## Check if gcloud is installed
if ! command -v gcloud &> /dev/null
then
    echo "gcloud is not installed. Please install it before continuing."
    exit 1
fi

source $CONFIG_FILE

if [ -z "$FOLDER" ]; then
    echo "Folder not specified"
    exit 1
fi
if [ -z "$COMMAND" ]; then
    echo "Command not specified"
    exit 1
fi

### Prepare some Data
if [ ! -z "$EXTRA_EXCLUDES" ]; then
  FOLDERS_TO_EXCLUDE="${DEFAULT_EXCLUDES},${EXTRA_EXCLUDES}"
else
  FOLDERS_TO_EXCLUDE="${DEFAULT_EXCLUDES}"
fi
EXLUDE_STRING=""
# Iterate over comma sepearted list of folders to exclude and add to the exclude EXLUDE_STRING
IFS=',' read -r -a array <<< "$input"
for FOLDER_TO_EXCLUDE in "${array[@]}"; do
        EXLUDE_STRING="$EXLUDE_STRING --exclude=$FOLDER_TO_EXCLUDE"
done
## Check on status of the instance
INSTANCE_STATUS=$(gcloud compute instances describe $COMPUTE_INSTANCE_NAME --format="get(status)")
if [ "$INSTANCE_STATUS" != "RUNNING" ]; then
    ### Start the instance
    gcloud compute instances start "$COMPUTE_INSTANCE_NAME"

    ### Wait for the instance to be ready
    while ! gcloud compute ssh $COMPUTE_INSTANCE_NAME --command "echo 'ready'" ; do
      echo "Waiting for instance to be ready"
      sleep 10
    done
else
    echo "Instance is already running. Perhaps someone else is using it?"
    # Ensure the user wants to continue
    read -p "Do you want to continue? [y/N] " choice
    case "$choice" in
        y|Y ) echo "Continuing...";;
        n|N ) exit 1;;
        * ) echo "Invalid input. Please enter 'y' or 'n'";;
    esac
fi

### Need to figure accounts
ALL_INFO=$(gcloud compute instances describe $COMPUTE_INSTANCE_NAME --format="yaml(name, status, disks, metadata)")
POSSIBLE_ACCOUNTS=$(printf %s\n "$ALL_INFO" | yq -r '.metadata.items.[] | select(.key == "ssh-keys") | .value ')
IFS=$'\n' read -r -d '' -a options_array <<< "$POSSIBLE_ACCOUNTS"

# Do selection with some escape sequence pasas
# PS3="Select the account you want to use: "
echo -e "\033[1;32mAvailable accounts: \033[0m"
PS3=$'\e[1;32mPlease select an option: \e[0m'
select account in "${options_array[@]}"; do
    if [ -n "$account" ]; then
        echo -e "\033[1;32mYou selected:\033[0m $account"
        break
    else
        echo "Invalid selection. Try again."
    fi
done

#Select everything b4 the colon
USER=$(echo $account | cut -d: -f1)
echo -e "\033[1;32mUser:\033[0m $USER"

INSTANCE_DESCRIPTION=$(gcloud compute instances describe $COMPUTE_INSTANCE_NAME --format="yaml(name,status,disks, networkInterfaces[0].accessConfigs[0].natIP)")

IP_ADDRESS=$(echo "$INSTANCE_DESCRIPTION" | yq -r '.networkInterfaces[0].accessConfigs[0].natIP')
echo -e "\033[1;32m✅ Instance is ready\033[0m"
echo -e "\033[1;32m✅ Instance IP Address: $IP_ADDRESS\033[0m"

echo "Instance is ready"
echo $INSTANCE_DESCRIPTION

echo "Will start the synchronization.."

### rsync the flake.nix file to make sure it is up to date
scp ./flake.nix "${USER}@${IP_ADDRESS}:~/flake.nix"
RSYNC_COMMAND="rsync -avz ./flake.nix ${USER}@${IP_ADDRESS}:/${FOLDER}/"
INSTALL_NIX_COMMAND="
if ! command -v nix-shell &> /dev/null
then
    echo 'Nix is not installed. Installing Nix...'
    curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install --no-confirm
    echo 'experimental-features = nix-command flakes' | sudo tee /etc/nix/nix.conf >> /dev/null
else
    echo 'Nix is already installed.'
fi
"
ssh "${USER}@${IP_ADDRESS}" "$INSTALL_NIX_COMMAND"

INSTALL_FLAKE_COMMAND="nix develop -c rsync --version"
ssh "${USER}@${IP_ADDRESS}" "$INSTALL_FLAKE_COMMAND"

echo "$EXCLUDE_STRING"
echo "$FOLDER"
rsync -avz "$EXLUDE_STRING" $FOLDER/ "${USER}@${IP_ADDRESS}:~"
 
# Finally SSH into it
ssh "${USER}@${IP_ADDRESS}" "nix develop"
