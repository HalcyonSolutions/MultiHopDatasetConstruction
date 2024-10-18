# Introduction
Triplet Creation is a comprehensive suite of scripts and modules designed to construct a Knowledge Graph dataset that integrates seamlessly with both Wikidata and Neo4j. These tools enable the extraction, conversion, and management of data from diverse sources, including Freebase and Jeopardy, facilitating the creation of enriched RDF triples and efficient graph-based queries.

## Freebase to Wikidata (freebase_2_wikidata)
This module takes the MDI from FB15K-237 and converts it into RDF values, creating a set of 15,000 nodes to be queried in Wikidata for additional information and relationships.

Note: For Part I, download [fb2w.nt.gz](https://developers.google.com/freebase); otherwise, proceed to Part II.

## Jeopardy to Wikidata (jeopardy_2_wikidata)
Extracts named entities from Jeopardy questions and answers, creating a set of nodes that can be queried in Wikidata for further information and relationships.

Note: This code utilizes the [Jeopardy Dataset](https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions)

## FB-Wiki Triplet Extraction (fbwiki)
These scripts facilitate the extraction and creation of RDF triples within FB-Wiki. They involve web scraping Wikidata and processing both the extracted triples and their individual components.

### Generating Your Own Wikidata-Compatible Dataset
To generate your own Wikidata-compatible dataset, start by creating a `.txt` file containing your desired entity nodes, and then run the following scripts in sequence (where `files` denote input files and `[files]` represent output files):
1. fbwiki_triplet_creation.py --entity-list-path `your-entity-nodes.txt` --output-path `[first-set-of-triplets-path.txt]`
2. fbwiki_triplet_process.py --first-stage True --primary-triplet-path `first-set-of-triplets-path.txt` --processed-triplet-output `[processed-triplets-path.txt]` --missing-nodes-output `[intermediate-nodes-path.txt]`
3. fbwiki_triplet_creation.py --entity-list-path `intermediate-nodes-path.txt` --output-path `[second-set-of-triplets-path.txt]`
4. fbwiki_triplet_process.py --first-stage False --missing-triplets-path `second-set-of-triplets-path.txt` --processed-triplet-output `processed-triplets-path.txt` --final-nodes-output `[your-final-node-set-path.txt]` --final-triplet-output `[your-final-triplets-path.txt]`
5. fbwiki_triplet_split.py --triplet-file-path `your-final-triplets-path.txt` --train-file-path `[train-path.txt]` --test-file-path `[test-path.txt]` --valid-file-path `[valid-path.txt]`
6. fbwiki_entity_data_retrieval.py --input-set-path `your-final-triplets-path.txt` --output-csv-path `[your-final-nodes-data-path.csv]`

For more details, refer to each script for specific instructions and configurations.

## Graph Management (graph)
This module manages and queries the FB-Wiki Graph dataset in Neo4j. It provides functionality for building the graph, querying nodes and their neighborhoods, and extracting paths of a specified length between existing nodes.

Ensure that the `./config/configs.ini` file is properly configured with the necessary details for connecting to Neo4j.

### Graph Path Extraction (graph_extract_paths)
This script creates a dataset of graph paths.

To run the script, see the example below:
```
python graph_extract_paths.py --min-hops 2 --max-hops 2 --total-paths 20000 --num-workers 10 --use-filter False --use-prune True --dataset-folder ./data/multi_hop/ --dataset-prefix 2 --dataset-suffix prune --use-rand-path True --reload-set False
```
The example above creates a dataset of 20,000 paths with a minimum and maximum of 2 hops, removing any non-informative relationships from paths between the head and tail nodes. It randomly selects paths from the valid set and uses 10 threads to speed up the process. The created dataset is saved to `./data/multi_hop/2_hop_prune.csv`.

If the program is halted at some point, set `--reload-set True` for the program to continue from its previous checkpoint.
