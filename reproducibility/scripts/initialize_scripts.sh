# Will initialize the environmetn in the shellHook

## Check if we have all the necessary environment variables
# Ask for password to use henceforth
read -s -p "Enter the password for the encrypted files: " PASSWORD

# First check for the .gpg files. Decrypt them all in place
LIST_OF_GPG_FILES=$(find . -name "*.gpg")
# Decrypt them all in place
for file in $LIST_OF_GPG_FILES; do
    name_without_gpg=$(echo $file | sed 's/\.gpg//g')
    if [ ! -f "$name_without_gpg" ]; then
      echo -e "\033[0;33m::Decrypting $file to $name_without_gpg\033[0m"
      RESULT=$(gpg --batch --yes --pinentry-mode loopback --passphrase "$PASSWORD" --decrypt $file )
      if [ $? -ne 0 ]; then
        echo -e "\033[0;31m::Decryption failed. Please check your gpg key and try again\033[0m"
        exit 1
      fi
      echo "$RESULT" > $name_without_gpg

      # If We have bucket_servacc.json then we use it to auth 
      base_name=$(basename $name_without_gpg)
      if [ "$base_name" == "bucket_servacc.json" ]; then # or use service_account.json
        echo -e "\033[0;33m::Found service account ${name_without_gpg}. Activating service account\033[0m"
        gcloud auth activate-service-account --key-file="$name_without_gpg" &> /dev/null
        ACCOUNT_NAME=$(cat "$name_without_gpg" | jq -r '.client_email' 2>/dev/null)
        if [ -n "$ACCOUNT_NAME" ] && [ "$ACCOUNT_NAME" != "null" ]; then
          echo -e "\033[0;32m::Activated service account $ACCOUNT_NAME\033[0m"
          gcloud config set account $ACCOUNT_NAME &> /dev/null
          export ACCOUNT_NAME=$(gcloud config get-value account 2>/dev/null)
        else
          echo -e "\033[0;31m::Failed to extract account name from service account JSON\033[0m"
        fi
      fi
    fi
done

