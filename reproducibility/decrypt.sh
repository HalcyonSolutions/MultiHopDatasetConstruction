#! /bin/bash

# Any files with gpg in this folder will be decrypted
FILES_TO_DECRYPT=$(find . -name "*.gpg")

# Ask users for password
echo "Enter password:"
read -s PASSWORD

for FILE_TO_DECRYPT in $FILES_TO_DECRYPT; do
    echo "Decrypting $FILE_TO_DECRYPT. DO NOT ADD TO REPO"
    # Remove gpg extension
    OUTP_FILE=${FILE_TO_DECRYPT%.gpg}
    gpg --decrypt --batch --passphrase "$PASSWORD" --output "$OUTP_FILE" "$FILE_TO_DECRYPT"
    echo "Removing $FILE_TO_DECRYPT."
    rm "$FILE_TO_DECRYPT"
done
