# Will initialize the environmetn in the shellHook

## Check if we have all the necessary environment variables

# First check for the .gpg files. Decrypt them all in place
LIST_OF_GPG_FILES=$(find . -name "*.gpg")
# Decrypt them all in place
for file in $LIST_OF_GPG_FILES; do
    name_without_gpg=$(echo $file | sed 's/\.gpg//g')
    if [ ! -f "$name_without_gpg" ]; then
      echo -e "\033[0;33m::Decrypting $file to $name_without_gpg\033[0m"
      RESULT=$(gpg --decrypt $file)
      if [ $? -ne 0 ]; then
        echo -e "\033[0;31m::Decryption failed. Please check your gpg key and try again\033[0m"
        exit 1
      fi
      echo "$RESULT" > $name_without_gpg
    fi
done

if [ ! -f "service_account.json" ]; then
  echo -e "\033[0;31m::No service account found. Please create one and try again\033[0m"
  exit 1
fi
gcloud auth activate-service-account --key-file=service_account.json &> /dev/null
ACCOUNT_NAME=$(cat service_account.json | jq -r '.client_email')
echo -e "\033[0;32m::Activated service account $ACCOUNT_NAME\033[0m"
gcloud config set account $ACCOUNT_NAME
export ACCOUNT_NAME=$(gcloud config get-value account)

