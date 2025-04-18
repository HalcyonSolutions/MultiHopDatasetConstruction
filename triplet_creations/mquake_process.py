"""
Sample code to extract the entity sets from mquake to further extract triplets

TODO: Remove (CF) and (T) edits from MQuaKE triplets. 
TODO: Enforce original triplets on MQuakE KG.
"""

import json
import pandas as pd
from utils.basic import load_json, sort_by_qid

files = ["./data/MQuAKE-CF-3k-v2.json", "./data/MQuAKE-CF.json", "./data/MQuAKE-T.json"] 

entities = set()
relations = set()

for file_path in files:
    data = load_json(file_path)

    for case in data:
        for triple_set_name in ["triples", "new_triples", "edit_triples"]:
            triple_set = case.get("orig", {}).get(triple_set_name, [])
            for head, rel, tail in triple_set:
                if head[0] == "Q":
                    entities.add(head)
                if tail[0] == "Q":
                    entities.add(tail)
                if rel[0] == "P":
                    relations.add(rel)

print("Total unique entities:", len(entities))
print("Total unique relations:", len(relations))


# Convert the list to a DataFrame
node_df = pd.DataFrame(entities, columns=['Node'])
node_df = sort_by_qid(node_df, column_name = 'Node')
node_df.to_csv('./data/nodes_mquake.txt', index=False, header=False)