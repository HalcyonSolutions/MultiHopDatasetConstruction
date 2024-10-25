# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:45:07 2024

@author: Eduin Hernandez

Summary:
This package provides classes and methods to create, update, and manage a Freebase-Wikidata
 Hybrid graph in Neo4j. It enables efficient creation of nodes and relationships, extraction
 of neighborhood paths, and filtering relationships based on node categories.

Core functionalities:
- **Graph Creation and Update**: Create and update nodes, relationships, and node properties in a Neo4j database.
- **Path Extraction**: Extract paths between nodes, including multi-hop relationships and random paths.
- **Relationship Filtering**: Manage and filter relationships based on categories to support complex queries.
"""

import ast
import pandas as pd
from tqdm import tqdm

from neo4j import GraphDatabase

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Dict, List, Tuple


from utils.basic import load_pandas

#------------------------------------------------------------------------------
'FB-Wiki Graph'
class FbWikiGraph():
    """
    This class is used to create and update a Freebase-Wikidata Hybrid graph in Neo4j. 
    It provides methods for creating nodes, relationships, updating node information, and querying the graph.
    """
    def __init__(self, uri:str, user:str, password:str, database: str = "relhierarchy") -> None:
        """
        Initializes the FbWikiGraph class with Neo4j connection details and database information.
        
        Args:
            uri (str): The URI for the Neo4j instance.
            user (str): The username for Neo4j authentication.
            password (str): The password for Neo4j authentication.
            database (str): The Neo4j database to connect to (default is "relhierarchy").
        """

        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
    
    # Function to clear all data from the graph
    def clear_graph(self, tx) -> None:
        """
        Clears all data from the graph by detaching and deleting all nodes.
        
        Args:
            tx: A Neo4j transaction object used to run the query.
        """
        tx.run("MATCH (n) DETACH DELETE n")

    def get_drive(self) -> GraphDatabase:
        """
        Returns a Neo4j driver instance to establish a connection with the database.
        
        Returns:
            Neo4j driver: A driver object to connect to Neo4j.
        """
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    # Function to create nodes for each RDF title mapping
    def _create_nodes(self, tx, rdf_valid: List[str]) -> None:
        """
        Creates nodes in the graph for a given list of RDF identifiers.
        
        Args:
            tx: A Neo4j transaction object.
            rdf_valid (list): A list of valid RDF identifiers for node creation.
        """
        for rdf in tqdm(rdf_valid, desc='Creating nodes'):
            tx.run("CREATE (:Node {RDF: $rdf})", rdf=rdf)
        
    def _create_new_nodes(self, tx, rdf_valid: List[str]) -> None:
        """
        Creates new nodes in the graph, ensuring no duplicates are created.
        
        Args:
            tx: A Neo4j transaction object.
            rdf_valid (list): A list of valid RDF identifiers for node creation.
        """
        for rdf in tqdm(rdf_valid, desc='Checking and creating nodes'):
            query = (
                """
                MERGE (n:Node {RDF: $rdf})
                ON CREATE SET n.RDF = $rdf
                """
            )
            tx.run(query, rdf=rdf)
    
    def _create_link_batch(self, batch: List[Tuple[str, dict, str]]):
        """
        Creates relationships between nodes in batches.
        
        Args:
            batch (List[Tuple[str, dict, str]]): A batch of tuples, each containing (rdf_from, prop, rdf_to).
        """
        driver = self.get_drive()
        with driver.session(database=self.database) as session:
            for rdf_from, prop, rdf_to, rel_type in batch:
                # Create the query string with rel_type inserted directly
                query = (
                f"""
                MATCH (a:Node {{RDF: $rdf_from}}), (b:Node {{RDF: $rdf_to}})
                MERGE (a)-[r:{rel_type} {{
                    Title: $prop.Title,
                    Property: $prop.Property,
                    Description: $prop.Description,
                    Alias: $prop.Alias
                }}]->(b)
                """
                )
                # Run the query with the parameters
                session.run(query, rdf_from=rdf_from, rdf_to=rdf_to, prop=prop)
                
        driver.close()

    def _update_node_batch(self, info_batch: List[dict]):
        """
        Updates information of nodes in batches within the graph.
        
        Args:
            info_batch (list): A list of dictionaries, each containing node information to update.
        """
        driver = self.get_drive()
        with driver.session(database=self.database) as session:
            # Create a Cypher query that processes a batch of nodes
            query = """
            UNWIND $info_batch AS info
            MATCH (n:Node {RDF: info.RDF})
            SET n.Title = info.Title,
                n.Description = info.Description, 
                n.MDI = info.MDI, 
                n.URL = info.URL, 
                n.Alias = info.Alias,
                n.Forwarding = info.Forwarding
            """
            # Run the batch update query
            session.run(query, info_batch=info_batch)
            
        driver.close()

    
    #--------------------------------------------------------------------------
    'Functions to Initialize and Modify Nodes + Relationships'
    # Function to process the txt files and create the graph in Neo4j
    def create_graph(self, rdf_valid: List[str]) -> None:
        """
        Clears the existing graph and creates new nodes for a given list of RDF identifiers.
        
        Args:
            rdf_valid (list): A list of valid RDF identifiers for node creation.
        """
        driver = self.get_drive()
        
        with driver.session(database=self.database) as session:
            # Clear the graph before adding new data
            session.execute_write(self.clear_graph)
            # Create nodes for each RDF title mapping
            session.execute_write(self._create_nodes, rdf_valid)
        driver.close()
        
    def create_new_nodes(self, rdf_valid: List[str]) -> None:
        """
        Creates new nodes in the graph without clearing the existing data.
        
        Args:
            rdf_valid (list): A list of valid RDF identifiers for node creation.
        """
        
        driver = self.get_drive()
        
        with driver.session(database=self.database) as session:
            # Create nodes for each RDF title mapping
            session.execute_write(self._create_new_nodes, rdf_valid)
        driver.close()
    
    def create_link_between_nodes(self, relation_map: pd.DataFrame, file_name: str,
                                  max_workers: int = 15, batch_size: int = 100) -> None:
        """
        Creates relationships between nodes in batches using threading and batch processing.
        
        Args:
            relation_map (pd.DataFrame): A DataFrame mapping properties to relationships.
            file_name (str): The file containing triplet data.
            batch_size (int): Number of links to process in each batch (default is 100).
            max_workers (int): Maximum number of threads to use (default is 15).
        """
    
        # Load the triplets from file
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    
        # Prepare the batch data
        total_tasks = len(lines)
        batch = []
        futures = []
    
        # ThreadPoolExecutor to manage threads
        with tqdm(total=total_tasks, desc='Creating links') as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, line in enumerate(lines):
                    rdf_from, prop, rdf_to = line.strip().split()
                    prop_info = relation_map[relation_map['Property'] == prop].iloc[0].to_dict()
                    
                    batch.append((rdf_from, prop_info, rdf_to, prop_info['Property']))
    
                    # If batch size is reached, submit the batch for processing
                    if len(batch) >= batch_size:
                        futures.append(executor.submit(self._create_link_batch, batch.copy()))
                        batch.clear()
    
                    # If it's the last iteration and there's any remaining data
                    if idx == total_tasks - 1 and batch:
                        futures.append(executor.submit(self._create_link_batch, batch.copy()))
                        batch.clear()
    
                    # Process completed futures and update the progress bar
                    if len(futures) >= max_workers:
                        for future in as_completed(futures):
                            try:
                                future.result()  # Raises exceptions if any
                            except Exception as e:
                                print(f"Error creating link: {e}")
                            finally:
                                pbar.update(batch_size)
                        futures.clear()
    
                # Wait for any remaining futures to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error creating link: {e}")
                    finally:
                        pbar.update(batch_size)

    def update_nodes_base_information(self, rdf_info_map: pd.DataFrame,
                                      max_workers: int = 15, batch_size: int = 100) -> None:
        """
        Updates the base information of nodes in Neo4j using batch processing and threading.
        
        Args:
            rdf_info_map (pd.DataFrame): A DataFrame containing node information to update.
            max_workers (int): Maximum number of threads to use (default is 15).
            batch_size (int): Number of nodes to include in each batch update (default is 100).
        """
        
        total_tasks = len(rdf_info_map)
        
        # Create the progress bar for tracking the total task completion
        with tqdm(total=total_tasks, desc='Updating nodes') as pbar:
            # Using ThreadPoolExecutor to manage a pool of threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                info_batch = []
    
                # Submit tasks in batches
                for idx, (_, info) in enumerate(rdf_info_map.iterrows()):
                    
                    info_batch.append(info.to_dict())
    
                    # When batch size is reached, submit the batch
                    if len(info_batch) >= batch_size:
                        futures.append(executor.submit(self._update_node_batch, info_batch.copy()))
                        info_batch.clear()  # Clear the batch
    
                    # If this is the last iteration and there are remaining items in the batch
                    if idx == total_tasks - 1 and info_batch:
                        futures.append(executor.submit(self._update_node_batch, info_batch.copy()))
                        info_batch.clear()
    
                    # Process completed futures and update the progress bar
                    if len(futures) >= max_workers:
                        for future in as_completed(futures):
                            try:
                                future.result()  # Raises exceptions if any
                            except Exception as e:
                                print(f"Error updating node: {e}")
                            finally:
                                # Update progress bar by batch size
                                pbar.update(batch_size)
                        futures.clear()
    
                # Wait for any remaining futures to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error updating node: {e}")
                    finally:
                        pbar.update(batch_size)

    def update_node_category(self, rdf_info_map: pd.DataFrame) -> None:
        #TODO: Check and fix after the new update
        """
        Updates the category information for nodes in the graph.
        
        Args:
            rdf_info_map (pd.DataFrame): A DataFrame containing category information for nodes.
        """
        
        driver = self.get_drive()
        with driver.session(database=self.database) as session:
            for _, info in tqdm(rdf_info_map.iterrows(), desc='Updating nodes', total=len(rdf_info_map)):
                query = (
                    """
                    MATCH (n:Node {RDF: $rdf})
                    UNWIND $categories_values AS cv
                    SET n.Category = cv.Category,
                        n.has_category = cv.has_category
                    """
                )
                session.run(query, rdf=info['RDF'],
                            categories_values=info.to_dict()
                            )
        driver.close()
        
    #--------------------------------------------------------------------------
    'Functions to Extract Nodes'
    def match_node(self, rdf: str) -> Dict:
        """
        Finds and returns the information of a node based on its RDF identifier.
        
        Args:
            rdf (str): The RDF identifier of the node to look up.
        
        Returns:
            dict: The information of the node if found, otherwise an empty dictionary.
        """
        assert rdf.startswith('Q'), "rdf must be a vlid RDF identifier starting with 'Q'"
        
        driver = self.get_drive()
        
        result = {}
        with driver.session(database=self.database) as session:
            query = ("MATCH (a:Node {RDF: $rdf})"
                     "RETURN a")
            nodes = session.run(query, rdf=rdf)
            result = nodes.single().data()['a'] if nodes.peek() else {} 
            
        driver.close()
        return result
    
    def match_related_nodes(self, rdf: str, direction: str = 'any',
                            rdf_only: bool = False) -> Tuple[List[any], List[any]]:
        """
        Finds nodes and relationships connected to a given node by its RDF identifier.
        
        Args:
            rdf (str): The RDF identifier of the node to search.
            direction (str): The direction of relationships to match ('any', '<-', '->').
            rdf_only (bool): If True, returns only the RDF identifiers of nodes and relationships. 
        
        Returns:
            tuple: A tuple containing lists of nodes and relationships connected to the input node.
        """
        assert rdf.startswith('Q'), "rdf must be a valid RDF identifier starting with 'Q'"
        assert direction in {'any', '<-', '->'}, "direction must be one of 'any', '<-', or '->'"

        driver = self.get_drive()

        nodes, rels = [], []
        try:
            with driver.session(database=self.database) as session:
                # Choose the query pattern based on the direction
                if direction == 'any':
                    query = ("MATCH (n:Node {RDF: $rdf})-[r]-(connected)")
                elif direction == '<-':
                    query = ("MATCH (n:Node {RDF: $rdf})<-[r]-(connected)")
                elif direction == '->':
                    query = ("MATCH (n:Node {RDF: $rdf})-[r]->(connected)")

                query += " RETURN r, connected"

                result = session.run(query, rdf=rdf)

                if result.peek():
                    for record in result:
                        if rdf_only:
                            nodes.append(record['connected']['RDF'])
                            rels.append(record['r']['Property'])
                        else:
                            nodes.append(dict(record['connected']))
                            rels.append(dict(record['r']))
        finally:
            driver.close()

        return nodes, rels

    def find_relationships(self, rdf_node_A: str, rdf_node_B: str, direction: str = 'any',
                            rdf_only: bool = False) -> List[any]:
        """
        Finds relationships between two nodes identified by their RDF identifiers.
        
        Args:
            rdf_node_A (str): The RDF identifier of the start node.
            rdf_node_B (str): The RDF identifier of the end node.
            direction (str): The direction of relationships to match ('any', '<-', '->').
            rdf_only (bool): If True, returns only the RDF identifiers of relationships.
        
        Returns:
            list: A list of relationships between the two nodes.
        """
        assert rdf_node_A.startswith('Q'), "rdf_node_A must be a valid RDF identifier starting with 'Q'"
        assert rdf_node_B.startswith('Q'), "rdf_node_B must be a valid RDF identifier starting with 'Q'"
        assert direction in {'any', '<-', '->'}, "direction must be one of 'any', '<-', or '->'"
    
        driver = self.get_drive()
    
        relationships = []
        try:
            with driver.session(database=self.database) as session:
                # Choose the query pattern based on the direction
                if direction == 'any':
                    query = ("MATCH (n:Node {RDF: $rdf_node_A})-[r]-(m:Node {RDF: $rdf_node_B})")
                elif direction == '<-':
                    query = ("MATCH (n:Node {RDF: $rdf_node_A})<-[r]-(m:Node {RDF: $rdf_node_B})")
                elif direction == '->':
                    query = ("MATCH (n:Node {RDF: $rdf_node_A})-[r]->(m:Node {RDF: $rdf_node_B})")
    
                query += " RETURN r"
    
                result = session.run(query, rdf_node_A=rdf_node_A, rdf_node_B=rdf_node_B)
    
                if result.peek():
                    for record in result:
                        if rdf_only:
                            relationships.append(record['r']['Property'])
                        else:
                            relationships.append(dict(record['r']))
        finally:
            driver.close()
    
        return relationships
    
    def find_neighborhood(self, rdf_list: List[str], max_degree: int = 1, limit: int = 0,
                          relationship_types: List[str] = None, rdf_only: bool = False,
                          rand: bool = False) -> List[List[any]]:
        """
        Retrieves the neighborhood of a list of nodes (identified by RDF) in the graph, up to a specified degree.
        
        Args:
            rdf_list (List[str]): A list of RDF node identifiers (must start with 'Q').
            max_degree (int): Maximum number of hops (relationship depth) to search in the neighborhood (default is 1).
            limit (int): The maximum number of results to return. If set to 0 or negative, no limit is applied (default is 0).
            relationship_types (List[str], optional): A list of specific relationship types to filter by. If None, all relationships are considered (default is None).
            rdf_only (bool): If True, returns only the RDF identifiers of the nodes. If False, returns full node information (default is False).
            rand (bool): If True, returns the results in random order. If False, the default ordering is used (default is False).
        
        Returns:
            List[List[any]]: A list of dictionaries representing the neighborhood nodes, or their RDF identifiers if `rdf_only` is set to True.
        """
        
        for r0 in rdf_list: assert r0.startswith('Q'), "rdf_list must be a valid RDF identifier starting with 'Q'"
        
        driver = self.get_drive()
        
        neighborhood = []
        try:
            with driver.session(database=self.database) as session:
                nodes_list = ", ".join(f"'{item}'" for item in rdf_list) 
                
                relationship_filter = ' | '.join(relationship_types) if relationship_types else ""
                relationship_part = f"[r:{relationship_filter} *..{max_degree}]"  if relationship_types else f"[*..{max_degree}]"
                limit_part = f"LIMIT {limit}" if (type(limit) == int and limit > 0) else ""
                rand_part = "ORDER BY rand()" if rand else ""
                
                query = (
                    f"""
                    MATCH (n)-{relationship_part}-(m)
                    WHERE n.RDF IN [{nodes_list}]
                    RETURN DISTINCT m AS nodes
                    {rand_part}
                    {limit_part}
                    """
                )
        
        
                result = session.run(query)
                
                if result.peek():
                    if rdf_only:
                        neighborhood = [record['nodes']['RDF'] for record in result]
                    else:
                        neighborhood = [dict(record['nodes']) for record in result]
        finally:
            driver.close()
        return neighborhood
    
    def find_path(self, rdf_start: str, rdf_end: str, min_hops: int = 2, max_hops: int = 3, limit: int = 1,
                  relationship_types: List[str] = None, noninformative_types: List[str] = [],
                  rdf_only: bool = False, rand: bool = False) -> List[Tuple[List[any], List[any]]]:
        #TODO: Check and fix after the new update
        """
        Finds multiple paths between two nodes in the graph, filtering by hop count, relationship types, and other options.
        
        Args:
            rdf_start (str): The RDF identifier of the start node.
            rdf_end (str): The RDF identifier of the end node.
            min_hops (int): Minimum number of hops (default is 2).
            max_hops (int): Maximum number of hops (default is 3).
            limit (int): Maximum number of paths to return (default is 1).
            relationship_types (list): A list of relationship types to include in the path search.
            noninformative_types (list): A list of relationship types to exclude from the path search.
            rdf_only (bool): If True, returns only RDF identifiers for nodes and relationships.
            rand (bool): If True, the paths are returned in random order.
        
        Returns:
            list: A list of tuples containing the nodes and relationships in each path found.
                - The first list contains the nodes in the path, with each node represented as a dictionary (or just the RDF identifier if `rdf_only` is True).
                - The second list contains the relationships in the path, with each relationship represented as a dictionary (or just the relationship property if `rdf_only` is True).
        """
        
        assert rdf_start.startswith('Q'), "rdf_start must be a valid RDF identifier starting with 'Q'"
        assert rdf_end.startswith('Q'), "rdf_end must be a valid RDF identifier starting with 'Q'"
    
        driver = self.get_drive()
        
        paths = []
        try:
            with driver.session(database=self.database) as session:
                # Construct the query with or without relationship types filtering
                relationship_filter = ' | '.join(relationship_types) if relationship_types else ""
                relationship_part = f"[r:{relationship_filter} * {min_hops}..{max_hops}]" if relationship_types else f"[*{min_hops}..{max_hops}]"
                
                # Prevents specific relationship types from showing in the paths
                noninformative_pruning = ", ".join(f"'{item}'" for item in noninformative_types) if noninformative_types else ""
                noninformative_part = f"WHERE NONE(rel IN relationships(path) WHERE type(rel) IN [{noninformative_pruning}])" if noninformative_types else ""
                
                # prevents cyclic nodes
                noncyclic = "AND " if noninformative_types else "WHERE "
                noncyclic += "ALL(node IN nodes(path) WHERE single(x IN nodes(path) WHERE x = node))"
                
                # randomization
                rand_part_a = "WITH path LIMIT 1000" if rand else ""
                rand_part_b = "ORDER BY rand()" if rand else ""
                
                limit_part = f"LIMIT {limit}" if (type(limit) == int and limit > 0) else ""

                query = (
                    f"""
                    MATCH path = (n {{RDF: $rdf_start}})-{relationship_part}-(m {{RDF: $rdf_end}})
                    {rand_part_a}
                    {noninformative_part}
                    {noncyclic}
                    RETURN nodes(path) AS nodes, relationships(path) AS relationships
                    {rand_part_b}
                    {limit_part}
                    """
                )
                
                result = session.run(query, rdf_start=rdf_start, rdf_end=rdf_end)
                
                if result.peek():
                    if rdf_only:
                        paths = [
                            (
                                [node['RDF'] for node in record['nodes']],
                                [rel['Property'] for rel in record['relationships']]
                            )
                            for record in result
                        ]
                    else:
                        paths = [
                            (
                                [dict(node) for node in record['nodes']],
                                [dict(rel) for rel in record['relationships']]
                            )
                            for record in result
                        ]
            
        finally:
            driver.close()
        return paths
#------------------------------------------------------------------------------
'Relationship Hierarchy Graph'
class RelHierGraph():
    """
    Graph Used to create and update the realtion hierarchy from the Freebase-Wikidata 
        Hybrid in Neo4j
    """
    def __init__(self, uri:str, user:str, password:str, database: str = "relhierarchy") -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
    
    # Function to clear all data from the graph
    def clear_graph(self, tx):
        tx.run("MATCH (n) DETACH DELETE n")

    def get_drive(self):
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    # Function to create nodes for each RDF title mapping
    def _create_nodes(self, tx, property_valid) -> None:
        for pid in tqdm(property_valid, desc='Creating nodes'):
            tx.run("CREATE (:Node {Property: $pid})", pid=pid)
        
    def _create_new_nodes(self, tx, property_valid: List[str]) -> None:
        for pid in tqdm(property_valid, desc='Checking and creating nodes'):
            query = (
                """
                MERGE (n:Node {Property: $pid})
                ON CREATE SET n.Property = $pid
                """
            )
            tx.run(query, pid=pid)
    
    #--------------------------------------------------------------------------
    'Functions to Initialize and Modify Nodes + Relationships'
    # Function to process the txt files and create the graph in Neo4j
    def create_graph(self, property_valid: List[str]) -> None:
        driver = self.get_drive()
        
        with driver.session(database=self.database) as session:
            # Clear the graph before adding new data
            
            session.execute_write(self.clear_graph)
            
            # Create nodes for each RDF title mapping
            session.execute_write(self._create_nodes, property_valid)
        driver.close()
        
    def create_new_nodes(self, property_valid: List[str]) -> None:
        driver = self.get_drive()
        
        with driver.session(database=self.database) as session:
            # Create nodes for each RDF title mapping
            session.execute_write(self._create_new_nodes, property_valid)
        driver.close()
    
    def update_nodes_base_information(self, relation_map: pd.DataFrame) -> None:
        driver = self.get_drive()
        with driver.session(database=self.database) as session:
            for _, info in tqdm(relation_map.iterrows(), desc='Updating nodes info', total=len(relation_map)):
                query = (
                    """
                    MATCH (n:Node {Property: $info.Property})
                    SET n.Title = $info.Title,
                        n.Property = $info.Property, 
                        n.Description = $info.Description,
                        n.Alias = $info.Alias
                    """
                )
                session.run(query, info=info.to_dict())
        driver.close()

    def create_link_between_nodes(self, relation_map: pd.DataFrame, file_name: str) -> None:
        driver = self.get_drive()
        with driver.session(database=self.database) as session:
            with open(file_name, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in tqdm(lines, desc="Creating links", position=0, total=len(lines)):
                    pid_from, prop, pid_to = line.strip().split()
                    prop = relation_map[relation_map['Property'] == prop].iloc[0].to_dict()
                    rel_type = prop['Title'].replace(" ", "_")
                    
                    query = (
                        f"""
                        MATCH (a:Node {{Property: $pid_from}}), (b:Node {{Property: $pid_to}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        """
                    )
                    session.run(query, pid_from=pid_from, pid_to=pid_to)
                    
        driver.close()
        
#------------------------------------------------------------------------------
    
class NodeRelationshipFilter():
    """
    A class to filter nodes and relationships in a Freebase-Wikidata hybrid 
    graph stored in Neo4j. This class is designed to load and manage relationships
    and nodes data, allowing for the retrieval and filtering of relationship types
    based on node categories and relationships to be used in Neo4j queries.
    """
    def __init__(self, rels_path: str, rels_filter_path: str, nodes_path: str) -> None:
        """
        Initializes the NodeRelFilter class by loading data from specified paths.
        
        Args:
            rels_path (str): Path to the CSV file containing relationship data.
            rels_filter_path (str): Path to the CSV file containing relationship filter data.
            nodes_path (str): Path to the CSV file containing node data.
        """
        self.rels_df        = load_pandas(rels_path)
        self.rel_filter_df  = load_pandas(rels_filter_path)
        self.nodes_df       = load_pandas(nodes_path)
        
        self.invalid_rels   = self.rels_df[self.rels_df['Non-Informative'] == False]['Property'].tolist()
        self.invalid_neo    = self.rels_df[self.rels_df['Non-Informative'] == False]['Neo4j'].tolist()

    def get_parents(self, node: List[str]) -> List[str]:
        """
        Retrieves the parent categories of a given node.
        
        Args:
            node (List[str]): A list containing the RDF identifier of the node to query.
        
        Returns:
            List[str]: A list of RDF identifiers representing the parent categories of the node.
                       If the node has no categories, it returns the node itself.
        """
        row = self.nodes_df.loc[self.nodes_df['RDF'] == node]
        if row['has_category'].iloc[0]: return ast.literal_eval(row['Category'].iloc[0])
        else: return [node]

    def parent_filters(self, parents: List[str]) -> List[str]:
        """
        Combines the relationship filters for a list of parent categories.
        
        Args:
            parents (List[str]): A list of RDF identifiers representing parent categories.
        
        Returns:
            List[str]: A list of relationship properties (column names) where the filter criteria are met.
        """
        # Initialize combined_row as a boolean series with False for all columns except 'RDF'
        combined_row = pd.Series(False, index=self.rel_filter_df.columns.drop('RDF'))
        
        # Iterate through the list `parents` and apply the OR operation across the relevant rows
        for rdf_value in parents:
            if rdf_value in self.rel_filter_df['RDF'].values:  # Check if the RDF value exists in the DataFrame
                rows = self.rel_filter_df.loc[self.rel_filter_df['RDF'] == rdf_value].drop(columns='RDF')
                combined_row |= rows.any(axis=0)  # Apply OR operation with each row
        
        # Extract the column names where the entry is True
        return combined_row.index[combined_row].tolist()
    
    def _rel2neo4j(self, rel_prop: List[str]) -> List[str]:
        """
        Maps relationship properties to their Neo4j equivalents.
        
        Args:
            rel_prop (List[str]): A list of relationship properties to map.
        
        Returns:
            List[str]: A list of Neo4j relationship types corresponding to the input properties.
        """
        df = pd.DataFrame(rel_prop, columns=['Property'])
        merged_df = pd.merge(df, self.rels_df, on='Property', how='left')
        return merged_df['Neo4j'].tolist()
    
    def nodes_rel_filters(self, start_node: List[str], end_node: List[str], remove_noninformative:bool = False) -> List[str]:
        """
        Filters the relationships between two nodes based on their parent categories
        and converts them to Neo4j relationship types.
        
        Args:
            start_node (List[str]): The RDF identifier of the start node.
            end_node (List[str]): The RDF identifier of the end node.
            remove_invalid (bool): Whether to remove noninformative relationships
        
        Returns:
            List[str]: A list of Neo4j relationship types that satisfy the filter criteria between the two nodes.
        """
        p0 = self.get_parents(start_node)
        p1 = self.get_parents(end_node)
        rels = self.parent_filters(p0 + p1)
        
        if remove_noninformative: rels = [item for item in rels if item not in self.invalid_rels]
        
        return self._rel2neo4j(rels)
    
    def get_noninform_rels(self, neo4j: bool = True) -> List[str]:    
        """
        Returns the list of noninformative relationships as a list. If neo4j is True, 
        it returns the string names compatible with Neo4j, otherwise as their Property
        Number.

        Args:
            neo4j (bool), optional: Whether to return the list of strings as a neo4j compatible.

        Returns:
            List[str]: A list containing the non-informative relationships.
        """
        if neo4j: return self.invalid_neo
        return self.invalid_rels
        