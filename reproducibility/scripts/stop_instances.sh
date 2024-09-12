# Will stop all instances linked to this user.
CURRENT_USER=$(gcloud config get-value account)
ALL_INSTANCES=$(gcloud compute instances list --filter="status=RUNNING" --format=json)
echo "$ALL_INSTANCES" > instances.json

# Check if empty. (Return [] if empty)
if [ "$ALL_INSTANCES" == "[]" ]; then
  # Show in yellow escape sequence
  echo -e "\033[0;33mNo instances found. Exiting.\033[0m"
  exit 0
fi

echo "Stopping all instances started by $CURRENT_USER"
for row in $(echo "${ALL_INSTANCES}" | jq -r '.[] | @base64'); do
    _jq() {
     echo ${row} | base64 --decode | jq -r ${1}
    }

    # echo "Going through the row: ${row}"

   INSTANCE=$(_jq '.name')
   ZONE=$(_jq '.zone')  # This returns  "https://www.googleapis.com/compute/v1/projects/<project_name>/zones/<desired_zone>"
   # CHECK: I am unsure this is robust enough. Might fail if url is not in the format above
   
   ITEMS=$(echo ${row} | base64 --decode | jq -r '.metadata.items[] | @base64')

   for row in "$ITEMS"; do
     _jq() {
       echo ${row} | base64 --decode | jq -r ${1}
     }
     key=$(_jq '.key')
     if [ "$key" == "creator" ]; then
       owner=$(_jq '.value')
       break
     fi
   done
    
   # Ensure that creator is the same as the current user
   if [ "$CURRENT_USER" != "$owner" ]; then
     continue
   fi

   echo -e "Stopping $INSTANCE at $ZONE for User $CURRENT_USER"
   gcloud compute instances stop $INSTANCE --zone=$ZONE
   # Check if returns an error
   if [ $? -ne 0 ]; then
     # Use red escape sequence to print in red
     echo -e "\033[0;31mStopping $INSTANCE at $ZONE failed. ❌\033[0m"
     exit 1
   else
     # Use green escape sequence to print in green
     echo -e "\033[0;32mStopping $INSTANCE at $ZONE succeeded. ✅\033[0m"
   fi
      
done
