"""
Created on 2025-04-24 

@author: Eduin Hernandez

Summary:
This package provides classes and methods to create, update, and manage a Simple
 graph in Neo4j. It enables efficient creation of nodes and relationships, extraction
 of neighborhood paths, and filtering relationships based on node categories.
 It assumes no name mapping is needed for the node and relationships. As the nodes are
 in the triplet, is how they'll apper in the graph.

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
#------------------------------------------------------------------------------
'FB Graph'
class SimpleGraph():
    """
    This class is used to create and update a Simple Graph in Neo4j. 
    It provides methods for creating nodes, relationships, updating node information, and querying the graph.
    """
    def __init__(self, uri:str, user:str, password:str, database: str = "simple") -> None:
        """
        Initializes the SimpleGraph class with Neo4j connection details and database information.
        
        Args:
            uri (str): The URI for the Neo4j instance.
            user (str): The username for Neo4j authentication.
            password (str): The password for Neo4j authentication.
            database (str): The Neo4j database to connect to (default is "fb15k").
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
    
    # Function to create nodes for each title title mapping
    def _create_nodes(self, tx, title_valid: List[str]) -> None:
        """
        Creates nodes in the graph for a given list of title (Machine Identifier).
        
        Args:
            tx: A Neo4j transaction object.
            title_valid (list): A list of valid title for node creation.
        """
        for title in tqdm(title_valid, desc='Creating nodes'):
            tx.run("CREATE (:Node {Title: $title})", title=title)
        
    def _create_new_nodes(self, tx, title_valid: List[str]) -> None:
        """
        Creates new nodes in the graph, ensuring no duplicates are created.
        
        Args:
            tx: A Neo4j transaction object.
            title_valid (list): A list of valid title for node creation.
        """
        for title in tqdm(title_valid, desc='Checking and creating nodes'):
            query = (
                """
                MERGE (n:Node {Title: $title})
                """
            )
            tx.run(query, title=title)
    
    def _create_link_batch(self, batch: List[Tuple[str, dict, str]]):
        """
        Creates relationships between nodes in batches.
        
        Args:
            batch (List[Tuple[str, dict, str]]): A batch of tuples, each containing (title_from, prop, title_to).
        """
        driver = self.get_drive()
        with driver.session(database=self.database) as session:
            for title_from, relation, title_to in batch:
                # print(f"{title_from} - {relation} -> {title_to}")
                # Create the query string with rel_type inserted directly
                query = (
                f"""
                MATCH (a:Node {{Title: $title_from}}), (b:Node {{Title: $title_to}})
                MERGE (a)-[r:{relation} {{Title: $relation}}]->(b)
                """
                )
                # Run the query with the parameters
                session.run(query, title_from=title_from, relation=relation, title_to=title_to)
                
        driver.close()

    def _update_node_batch(self, info_batch: List[dict]):
        """
        Updates information of nodes in batches within the graph.
        
        Args:
            info_batch (list): A list of dictionaries, each containing node information to update.
        """
        AssertionError("There is no node information in Simple Graph!!")
        return

    
    #--------------------------------------------------------------------------
    'Functions to Initialize and Modify Nodes + Relationships'
    # Function to process the txt files and create the graph in Neo4j
    def create_graph(self, title_valid: List[str]) -> None:
        """
        Clears the existing graph and creates new nodes for a given list of title.
        
        Args:
            title_valid (list): A list of valid title identifiers for node creation.
        """
        driver = self.get_drive()
        
        with driver.session(database=self.database) as session:
            # Clear the graph before adding new data
            session.execute_write(self.clear_graph)
            # Create nodes for each title title mapping
            session.execute_write(self._create_nodes, title_valid)
        driver.close()
        
    def create_new_nodes(self, title_valid: List[str]) -> None:
        """
        Creates new nodes in the graph without clearing the existing data.
        
        Args:
            title_valid (list): A list of valid title identifiers for node creation.
        """
        
        driver = self.get_drive()
        
        with driver.session(database=self.database) as session:
            # Create nodes for each title title mapping
            session.execute_write(self._create_new_nodes, title_valid)
        driver.close()
    
    def create_link_between_nodes(self, file_name: str,
                                  max_workers: int = 15, batch_size: int = 100) -> None:
        """
        Creates relationships between nodes in batches using threading and batch processing.
        
        Args:
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
                    title_from, relation, title_to = line.strip().split('\t')
                    
                    batch.append((title_from, relation, title_to))
    
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

    def update_nodes_base_information(self,
                                      max_workers: int = 15, batch_size: int = 100) -> None:
        """
        Updates the base information of nodes in Neo4j using batch processing and threading.
        
        Args:
            max_workers (int): Maximum number of threads to use (default is 15).
            batch_size (int): Number of nodes to include in each batch update (default is 100).
        """
        AssertionError("There is no node information in Simple Graph!!")
        return
        
    #--------------------------------------------------------------------------
    'Functions to Extract Nodes'
    def match_node(self, title: str) -> Dict:
        """
        Finds and returns the information of a node based on its title.
        
        Args:
            title (str): The title identifier of the node to look up.
        
        Returns:
            dict: The information of the node if found, otherwise an empty dictionary.
        """
        
        driver = self.get_drive()
        
        result = {}
        with driver.session(database=self.database) as session:
            query = ("MATCH (a:Node {Title: $title})"
                     "RETURN a")
            nodes = session.run(query, title=title)
            result = nodes.single().data()['a'] if nodes.peek() else {} 
            
        driver.close()
        return result
    
    def match_related_nodes(self, title: str, direction: str = 'any',
                            title_only: bool = False) -> Tuple[List[any], List[any]]:
        """
        Finds nodes and relationships connected to a given node by its title.
        
        Args:
            title (str): The title identifier of the node to search.
            direction (str): The direction of relationships to match ('any', '<-', '->').
            title_only (bool): If True, returns only the title of nodes and relationships. 
        
        Returns:
            tuple: A tuple containing lists of nodes and relationships connected to the input node.
        """
        assert direction in {'any', '<-', '->'}, "direction must be one of 'any', '<-', or '->'"

        driver = self.get_drive()

        nodes, rels = [], []
        try:
            with driver.session(database=self.database) as session:
                # Choose the query pattern based on the direction
                if direction == 'any':
                    query = ("MATCH (n:Node {Title: $title})-[r]-(connected)")
                elif direction == '<-':
                    query = ("MATCH (n:Node {Title: $title})<-[r]-(connected)")
                elif direction == '->':
                    query = ("MATCH (n:Node {Title: $title})-[r]->(connected)")

                query += " RETURN r, connected"

                result = session.run(query, title=title)

                if result.peek():
                    for record in result:
                        if title_only:
                            nodes.append(record['connected']['Title'])
                            rels.append(record['r']['Title'])
                        else:
                            nodes.append(dict(record['connected']))
                            rels.append(dict(record['r']))
        finally:
            driver.close()

        return nodes, rels

    def find_relationships(self, title_node_A: str, title_node_B: str, direction: str = 'any',
                            title_only: bool = False) -> List[any]:
        """
        Finds relationships between two nodes identified by their title.
        
        Args:
            titlenode_A (str): The title identifier of the start node.
            title_node_B (str): The title identifier of the end node.
            direction (str): The direction of relationships to match ('any', '<-', '->').
            title_only (bool): If True, returns only the title of relationships.
        
        Returns:
            list: A list of relationships between the two nodes.
        """
        assert direction in {'any', '<-', '->'}, "direction must be one of 'any', '<-', or '->'"
    
        driver = self.get_drive()
    
        relationships = []
        try:
            with driver.session(database=self.database) as session:
                # Choose the query pattern based on the direction
                if direction == 'any':
                    query = ("MATCH (n:Node {Title: $title_node_A})-[r]-(m:Node {Title: $title_node_B})")
                elif direction == '<-':
                    query = ("MATCH (n:Node {Title: $title_node_A})<-[r]-(m:Node {Title: $title_node_B})")
                elif direction == '->':
                    query = ("MATCH (n:Node {Title: $title_node_A})-[r]->(m:Node {Title: $title_node_B})")
    
                query += " RETURN r"
    
                result = session.run(query, title_node_A=title_node_A, title_node_B=title_node_B)
    
                if result.peek():
                    for record in result:
                        if title_only:
                            relationships.append(record['r']['Title'])
                        else:
                            relationships.append(dict(record['r']))
        finally:
            driver.close()
    
        return relationships
    
    def find_neighborhood(self, title_list: List[str], max_degree: int = 1, limit: int = 0,
                          relationship_types: List[str] = None, title_only: bool = False,
                          rand: bool = False) -> List[List[any]]:
        """
        Retrieves the neighborhood of a list of nodes (identified by title) in the graph, up to a specified degree.
        
        Args:
            title_list (List[str]): A list of title node identifiers
            max_degree (int): Maximum number of hops (relationship depth) to search in the neighborhood (default is 1).
            limit (int): The maximum number of results to return. If set to 0 or negative, no limit is applied (default is 0).
            relationship_types (List[str], optional): A list of specific relationship types to filter by. If None, all relationships are considered (default is None).
            title_only (bool): If True, returns only the title identifiers of the nodes. If False, returns full node information (default is False).
            rand (bool): If True, returns the results in random order. If False, the default ordering is used (default is False).
        
        Returns:
            List[List[any]]: A list of dictionaries representing the neighborhood nodes, or their title identifiers if `title_only` is set to True.
        """
        
        driver = self.get_drive()
        
        neighborhood = []
        try:
            with driver.session(database=self.database) as session:
                nodes_list = ", ".join(f"'{item}'" for item in title_list) 
                
                relationship_filter = ' | '.join(relationship_types) if relationship_types else ""
                relationship_part = f"[r:{relationship_filter} *..{max_degree}]"  if relationship_types else f"[*..{max_degree}]"
                limit_part = f"LIMIT {limit}" if (type(limit) == int and limit > 0) else ""
                rand_part = "ORDER BY rand()" if rand else ""
                
                query = (
                    f"""
                    MATCH (n)-{relationship_part}-(m)
                    WHERE n.Title IN [{nodes_list}]
                    RETURN DISTINCT m AS nodes
                    {rand_part}
                    {limit_part}
                    """
                )
        
        
                result = session.run(query)
                
                if result.peek():
                    if title_only:
                        neighborhood = [record['nodes']['Title'] for record in result]
                    else:
                        neighborhood = [dict(record['nodes']) for record in result]
        finally:
            driver.close()
        return neighborhood
    
    def find_path(self, title_start: str, title_end: str, min_hops: int = 2, max_hops: int = 3, limit: int = 1,
                  relationship_types: List[str] = None, noninformative_types: List[str] = [],
                  title_only: bool = False, rand: bool = False, can_cycle: bool = True) -> List[Tuple[List[any], List[any]]]:
        #TODO: Check and fix after the new update
        # !NOTE: Direction doesn't work in path finding for hops of 2 and on. Filter manually.
        """
        Finds multiple paths between two nodes in the graph, filtering by hop count, relationship types, and other options.
        
        Args:
            title_start (str): The title identifier of the start node.
            title_end (str): The title identifier of the end node.
            min_hops (int): Minimum number of hops (default is 2).
            max_hops (int): Maximum number of hops (default is 3).
            limit (int): Maximum number of paths to return (default is 1).
            relationship_types (list): A list of relationship types to include in the path search.
            noninformative_types (list): A list of relationship types to exclude from the path search.
            title_only (bool): If True, returns only title identifiers for nodes and relationships.
            rand (bool): If True, the paths are returned in random order.
        
        Returns:
            list: A list of tuples containing the nodes and relationships in each path found.
                - The first list contains the nodes in the path, with each node represented as a dictionary (or just the title identifier if `title_only` is True).
                - The second list contains the relationships in the path, with each relationship represented as a dictionary (or just the relationship property if `title_only` is True).
        """
    
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
                
                if can_cycle:
                    noncyclic = ""
                else:
                    # prevents cyclic nodes
                    noncyclic = "AND " if noninformative_types else "WHERE "
                    noncyclic += "ALL(node IN nodes(path) WHERE single(x IN nodes(path) WHERE x = node))"
                
                # randomization
                rand_part_a = "WITH path LIMIT 1000" if rand else ""
                rand_part_b = "ORDER BY rand()" if rand else ""
                
                limit_part = f"LIMIT {limit}" if (type(limit) == int and limit > 0) else ""

                query = (
                    f"""
                    MATCH (start {{Title: $title_start}}), (end {{Title: $title_end}})
                    MATCH path = (start)-{relationship_part}-(end)
                    {rand_part_a}
                    {noninformative_part}
                    {noncyclic}
                    WITH path, nodes(path) AS ns, relationships(path) AS rs
                    UNWIND range(0, size(rs) - 1) AS i
                    WITH path, ns, rs[i] AS rel, i
                    WITH path, ns, collect({{Title: rel.Title, direction: CASE WHEN startNode(rel) = ns[i] THEN '->' ELSE '<-' END}}) AS relationships
                    RETURN ns AS nodes, relationships
                    {rand_part_b}
                    {limit_part}
                    """
                )
                
                result = session.run(query, title_start=title_start, title_end=title_end)
                
                if result.peek():
                    if title_only:
                        paths = [
                            (
                                [node['Title'] for node in record['nodes']],
                                [(rel['Title'], rel['direction']) for rel in record['relationships']]
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