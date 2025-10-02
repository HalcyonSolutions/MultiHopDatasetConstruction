import argparse

from utils.basic import load_pandas, load_triplets
from utils.wikidata_v2 import update_relationship_data, update_entity_data

# TODO: Add option to compare vocabs and/or metadata files with triplets for checking the missing nodes/rels
# TODO: Add option to compare triplets with train/test/valid splits to check if there are missing nodes/rels
# TODO: Consider renaming file
# TODO: Move into test folder

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process entities data for Neo4j graph creation.")
    
    parser.add_argument('--entity-data-path', type=str, default='./data/vocabs/node_data_fb_wiki_v3.csv',
                        help='Path to the data of the entities.')
    parser.add_argument('--relationship-data-path', type=str, default='./data/metadata/relation_data_fb_wiki_v3.csv',
                        help='Path to the data of the relationship.')
    parser.add_argument('--triplets-data-path', type=str, default='./data/link_prediction/Fb-Wiki-V3/triplets.txt',
                        help='Path to the relationship between entities.')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    #--------------------------------------------------------------------------
    'Load the Data'
    node_data_map = load_pandas(args.entity_data_path)
    relation_map = load_pandas(args.relationship_data_path)

    triplet_df = load_triplets(args.triplets_data_path)
    nodes_set = set(triplet_df['head']) | set(triplet_df['tail'])
    rels_set = set(triplet_df['relation'])

    #--------------------------------------------------------------------------
    rel_data_set = set(relation_map['Property'].tolist())
    node_data_set = set(node_data_map['QID'].tolist())
    node_forwarding = set(node_data_map['Forwarding'].tolist())

    #--------------------------------------------------------------------------
    # print statements
    print('Number of unique nodes:', len(nodes_set))
    print('Number of unique relationships:', len(rels_set))
    print('Number of forward nodes:', len(node_forwarding))
    print('Forwarding nodes:\n', node_forwarding)
    #--------------------------------------------------------------------------
   
    missing_rels = rels_set - rel_data_set
    
    if missing_rels:
        print('Full relationship set:', len(rel_data_set))
        print('Number of Missing relationships:', len(missing_rels))
        print('Set of missing relationships:\n', missing_rels)
        rel_df = update_relationship_data(relation_map, missing_rels)
        rel_df.to_csv(args.relationship_data_path.replace('.csv','') + '_updated.csv', index=False)
        rels_data_set = set(rel_df['Property'].tolist())
        print('Number of unique relationships (Updated):', len(rels_data_set))
        print('Missing relationships: (Updated)', len(rels_set - rels_data_set))

    # --------------------------------------------------------------------------
    
    missing_nodes = nodes_set - node_data_set

    if missing_nodes:
        print('Full node set:', len(node_data_set))
        print('Number of Missing Nodes:', len(missing_nodes))
        print('Set of missing nodes:\n', missing_nodes)
        node_df = update_entity_data(node_data_map, missing_nodes)
        node_df.to_csv(args.entity_data_path.replace('.csv','') + '_updated.csv', index=False)
        nodes_data_set = set(node_df['QID'].tolist())
        print('Number of unique nodes (Updated):', len(nodes_data_set))
        print('Missing nodes: (Updated)', len(nodes_set - nodes_data_set))

