# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:51:05 2024

@author: Eduin Hernandez

Summary: This script processes Jeopardy questions to create a subgraph of neighboring
 nodes using Neo4j and FbWikiGraph. It includes argument parsing, data processing,
 and uploading to Neo4j. Key features include neighborhood extraction, subgraph creation,
 and saving relevant data for analysis.

"""
import argparse

from utils.basic import load_pandas, load_to_set, load_triplets
from utils.basic import save_triplets, save_set_pandas
from utils.basic import extract_literals, str2bool
from utils.configs import global_configs

from utils.fb_wiki_graph import FbWikiGraph

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process entities data for Neo4j graph creation.")
    
    # Stages
    parser.add_argument('--process-jeopardy', type=str2bool, default='False',
                        help='Whether to process the Jeopardy dataset.')
    parser.add_argument('--upload-jeopardy', type=str2bool, default='False',
                        help='Whether to upload the processed Jeopardy data to Neo4j.')
    
    # Jeopardy
    parser.add_argument('--jeopardy-path', type=str, default='./data/jeopardy_processed.csv',
                        help='Path to the Jeopardy CSV file containing the questions and answers RDF.')
    parser.add_argument('--jeopardy-questions-id', type=int, nargs='+',  
                        default=[81696, 2490, 20845, 26407, 33852, 34826, 53511, 72685, 83861, 26414],
                        help='List of question IDs from the Jeopardy dataset to process.')
    
    # Wikidata Info (Input)
    parser.add_argument('--entity-data-path', type=str, default='./data/node_data_fj_wiki.csv',
                        help='Path to the data of the entities.')
    parser.add_argument('--relationship-data-path', type=str, default='./data/relation_data_wiki.csv',
                        help='Path to the data of the relationship.')
    parser.add_argument('--triplets-data-path', type=str, default='./data/triplets_fj_wiki.txt',
                        help='Path to the relationship between entities.')
    
    # Neo4j
    parser.add_argument('--config-path', type=str, default='./configs/configs.ini',
                        help='Path to the configuration file for Neo4j connection.')
    parser.add_argument('--database-parent', type=str, default='fjwiki',
                        help='Name of the Neo4j database containing the full graph.')
    parser.add_argument('--database-subgraph', type=str, default='subgraph',
                        help='Name of the Neo4j database to use for the subgraph.')
    
    parser.add_argument('--max-workers', type=int, default=15, 
                        help='Number of workers to connect to Neo4j')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Batch size (rows of data) to use per worker')
    
    # Neighborhood Extraction
    parser.add_argument('--neighborhood-degree', type=int, default=1,
                        help='Degree of separation for the neighborhood extraction. For example, 1st degree includes parents and children, 2nd degree includes grandparents/grandchildren, etc.')
    parser.add_argument('--neighborhood-limit', type=int, default=2000,
                        help='Maximum number of nodes to extract in the neighborhood. Use 0 for no limit.')
    
    # Output
    parser.add_argument('--jeopardy-cherrypicked-path', type=str, default='./data/jeopardy_cherrypicked.csv',
                        help='Path to save the cherry-picked Jeopardy data.')
    parser.add_argument('--nodes-cherrypicked-path', type=str, default='./data/nodes_cherrypicked.txt',
                        help='Path to save the cherry-picked node list.')
    parser.add_argument('--nodes-data-cherrypicked-path', type=str, default='./data/node_data_cherrypicked.csv',
                        help='Path to save the cherry-picked node data.')
    parser.add_argument('--triplets-cherrypicked-path', type=str, default='./data/triplets_subgraph.txt',
                        help='Path to save the subgraph triplets data.')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    configs = global_configs('./configs/configs.ini')
    neo4j_parameters = configs['Neo4j']
    
    if args.process_jeopardy:
        
        'Prepare Jeopardy Questions'
        jeopardy = load_pandas(args.jeopardy_path)
        
        jeopardy_cherry_picked = jeopardy.iloc[args.jeopardy_questions_id]
        
        query_questions = extract_literals(jeopardy_cherry_picked['Question_RDF'], flatten=True)
        query_answers   = extract_literals(jeopardy_cherry_picked['Answer_RDF'], flatten=True)
        query_entities = list(set(query_questions + query_answers))
        
    
        #--------------------------------------------------------------------------
        'Prepare Wikidata Information'
        rdf_data_map = load_pandas(args.entity_data_path)
        triplets = load_triplets(args.triplets_data_path)
        
        rdf_data_map = rdf_data_map.fillna('')
        
        #--------------------------------------------------------------------------
        'Extract Neighborhood'
        parent_graph = FbWikiGraph(neo4j_parameters['uri'],
                                    neo4j_parameters['user'],
                                    neo4j_parameters['password'],
                                    database = args.database_parent)
        
        neighborhood = parent_graph.find_neighborhood(rdf_list=query_entities,
                                                      max_degree=args.neighborhood_degree,
                                                      limit=args.neighborhood_limit,
                                                      rand=True,
                                                      rdf_only=True)
        
        neighborhood += query_entities
        neighborhood = set(neighborhood)
        
        #--------------------------------------------------------------------------
        'Process the Subgraph'
        
        rdf_data_map = rdf_data_map[rdf_data_map['RDF'].isin(neighborhood)]
    
        triplets = triplets[(triplets['head'].isin(neighborhood)) &
                            (triplets['tail'].isin(neighborhood))]
        
        #--------------------------------------------------------------------------
        'Save Relevant Data'
        
        save_set_pandas(neighborhood, args.nodes_cherrypicked_path)
        jeopardy_cherry_picked.to_csv(args.jeopardy_cherrypicked_path)
        rdf_data_map.to_csv(args.nodes_data_cherrypicked_path)
        save_triplets(triplets, args.triplets_cherrypicked_path)

    
    if args.upload_jeopardy:
        #--------------------------------------------------------------------------
        'Load the Data'
        rdf_data_map = load_pandas(args.nodes_data_cherrypicked_path)
        relation_map = load_pandas(args.relationship_data_path)
        nodes = list(load_to_set(args.nodes_cherrypicked_path))
        
        relation_map = relation_map.fillna('')
        rdf_data_map = rdf_data_map.fillna('')

    
        'Update Subgraph'
        
        subgraph = FbWikiGraph(neo4j_parameters['uri'],
                               neo4j_parameters['user'],
                               neo4j_parameters['password'],
                               database = args.database_subgraph)
        
        subgraph.create_graph(list(nodes)) # Avoid using if a graph already exists as this will erase it
        
        subgraph.update_nodes_base_information(
            rdf_data_map,
            max_workers=args.max_workers, 
            batch_size=args.batch_size)
        
        subgraph.create_link_between_nodes(
            relation_map,
            args.triplets_cherrypicked_path,
            max_workers=args.max_workers,
            batch_size=args.batch_size)