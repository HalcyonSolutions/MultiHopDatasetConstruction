#!/bin/bash
set -euxo pipefail

# Mount your persistent disk (already attached via `gcloud compute instances attach-disk <instance-name> --disk=<blazegraph-hyper>)
mkdir -p /mnt/blazegraph-disk
chown -R ottersome:ottersome /mnt/blazegraph-disk
mount /dev/disk/by-id/google-blazegraph-hyper-part1 /mnt/blazegraph-disk

# Run chunk uploader in tmux so it survives startup
tmux new-session -d -s uploader "bash /mnt/blazegraph-disk/blazegraph/import-batches.sh"
