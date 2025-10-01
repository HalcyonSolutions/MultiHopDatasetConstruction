"""
Summary: Tests for Wikidata utility functions.

Run Command: python test/test_wikidata.py
"""

import sys
sys.path.append('.')

from utils.wikidata_v2 import get_thread_local_client, search_wikidata_relevant_id  # uses WIKIMEDIA_UA from config
from utils.wikidata_v2 import fetch_entity_details, fetch_head_entity_triplets
from utils.wikidata_v2 import fetch_relationship_details, fetch_relationship_triplet

if __name__ == '__main__':
    'Testing Parameter'
    query_text = 'Obama'
    query_entity = 'Q76'  # Barack Obama
    query_property = 'P39' # Position Held
    text_limit = 3
    triplet_limit = 10

    'Create Client'
    client = get_thread_local_client()
    print(dict(client.opener.addheaders)["User-Agent"])

    'Searching Entity by Name'
    entities = search_wikidata_relevant_id(query_text, topk=text_limit)
    print(f"Search Results for '{query_text}':")
    for ent in entities:
        print("=========")
        for key in ent:
            print(f"{key}: {ent[key]}")

    'Get Triplets for a given Entity'
    info = fetch_entity_details(query_entity)
    triplets, _, _ = fetch_head_entity_triplets(query_entity, limit=triplet_limit, mode='ignore')
    
    print(f"\n=========\n")
    for item in info:
        print(f"{item}: {info[item]}")
    print(f"Presenting the first {triplet_limit} triplets for {info['Title']} ({query_entity}):")
    for triplet in triplets:
        print(triplet)

    'Get Triplets for a given Property'
    info = fetch_relationship_details(query_property)
    
    print(f"\n=========\n")
    for item in info:
        print(f"{item}: {info[item]}")
    triplets, _, = fetch_relationship_triplet(query_property, limit=triplet_limit)
    print(f"Presenting the first {triplet_limit} triplets for {info['Title']} ({query_property}):")
    for triplet in triplets:
        print(triplet)