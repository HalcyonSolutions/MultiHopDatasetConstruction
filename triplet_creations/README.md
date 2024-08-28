# Introduction

## fbwiki_{}
This are the main codes used for the extraction and creation of the triplets in FB-Wiki which mainly involve Web-scrapping Wiki-Data and Processing both the triplets and their respective individual data.

## graph_{}
This is the main code for managing and querying the FB-Wiki Graph dataset in Neo4j. It includes building the graph, quering for nodes and their neighborhood, and extracting paths of set length between two existing nodes.
Make sure the file './config/configs.ini' is filled out with the necessary information for connecting to neo4j

### graph_extract_paths
This code create a dataset of path.
To run the code, see the example below:
```
python graph_extract_paths.py --min-hops 2 --max-hops 2 --total-paths 20000 --num-workers 10 --use-filter False --use-prune True --dataset-folder ./data/multi_hop/ --dataset-prefix 2 --dataset-suffix prune --use-rand-path True --reload-set False
```
The code above create a path dataset of size 20k with a min and max hop of 2, removing any non-informative relationships from the path between the head and tail pairing. It selects the path at random from the valid set. It uses 10 threads pools to speed the process. The created dataset is written to ./data/multi_hop/2_hop_prune.csv.

Should program be halted at some point, set `--reload-set True` for the program to continue from its previous checkpoint. 
