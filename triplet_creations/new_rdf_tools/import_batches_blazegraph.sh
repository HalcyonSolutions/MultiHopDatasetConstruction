#!/bin/bash
set -e

NAMESPACE="wikidata"
GRAPH="http://wikidata.org"
BLAZEGRAPH_JAR="/mnt/blazegraph-disk/blazegraph/blazegraph.jar"
CHUNKS_DIR="/mnt/blazegraph-disk/ttl_chunks"
LOG_FILE="/mnt/blazegraph-disk/blazegraph/uploaded.log"
PROPERTIES_FILE="/mnt/blazegraph-disk/blazegraph/RWStore.properties"

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

for file in "$CHUNKS_DIR"/*.ttl; do
    if grep -Fxq "$file" "$LOG_FILE"; then
        echo "Skipping already uploaded: $file"
        continue
    fi

    echo "Uploading $file"
    java -Xmx8g -cp "$BLAZEGRAPH_JAR" com.bigdata.rdf.store.DataLoader \
         -namespace "$NAMESPACE" \
         -defaultGraph "$GRAPH" \
         "$PROPERTIES_FILE" \
         "$file"

    echo "$file" >> "$LOG_FILE"
    sync
done

