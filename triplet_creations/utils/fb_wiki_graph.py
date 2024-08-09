# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:45:07 2024

@author: Eduin Hernandez
"""
from neo4j import GraphDatabase
import regex as re
from tqdm import tqdm

from typing import Dict, List, Tuple

class FbWikiGraph():
    """
    Graph Used to create and update Freebase-Wikidata Hybrid in Neo4j
    """
    def __init__(self, uri:str, user:str, password:str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
    
    # Function to clear all data from the graph
    def clear_graph(self, tx):
        tx.run("MATCH (n) DETACH DELETE n")

    def get_drive(self):
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    # Function to create nodes for each RDF title mapping
    def _create_nodes(self, tx, rdf_valid) -> None:
        for rdf in tqdm(rdf_valid, desc='Creating nodes'):
            tx.run("CREATE (:Node {RDF: $rdf})", rdf=rdf)
        
    def _create_new_nodes(self, tx, rdf_valid: List[str]) -> None:
        for rdf in tqdm(rdf_valid, desc='Checking and creating nodes'):
            query = (
                """
                MERGE (n:Node {RDF: $rdf})
                ON CREATE SET n.RDF = $rdf
                """
            )
            tx.run(query, rdf=rdf)
    
    #--------------------------------------------------------------------------
    'Functions to Initialize and Modify Nodes + Relationships'
    # Function to process the txt files and create the graph in Neo4j
    def create_graph(self, rdf_valid: List[str]) -> None:
        driver = self.get_drive()
        
        with driver.session() as session:
            # Clear the graph before adding new data
            session.execute_write(self.clear_graph)
            # Create nodes for each RDF title mapping
            session.execute_write(self._create_nodes, rdf_valid)
        driver.close()
        
    def create_new_nodes(self, rdf_valid: List[str]) -> None:
        driver = self.get_drive()
        
        with driver.session() as session:
            # Create nodes for each RDF title mapping
            session.execute_write(self._create_new_nodes, rdf_valid)
        driver.close()
    
    def create_link_between_nodes(self, relation_map: Dict, file_name: str) -> None:
        driver = self.get_drive()
        with driver.session() as session:
            with open(file_name, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in tqdm(lines, desc="Creating links", position=0):
                    rdf_from, prop, rdf_to = line.strip().split()
                    relation_name = re.sub('\W+', '_', relation_map[prop])  # Sanitize to create a valid relationship type
                    query = (
                        "MATCH (a:Node {RDF: $rdf_from}), (b:Node {RDF: $rdf_to}) "
                        "MERGE (a)-[r:" + relation_name + " {Title: $p_title, Property: $prop}]->(b)"
                    )
                    session.run(query, rdf_from=rdf_from, rdf_to=rdf_to,
                                p_title=relation_map[prop], prop=prop)
                    
        driver.close()
    
    def update_nodes_base_information(self, rdf_info_map: Dict) -> None:
        driver = self.get_drive()
        with driver.session() as session:
            for rdf, info in tqdm(rdf_info_map.items(), desc='Updating nodes'):
                query = (
                    """
                    MATCH (n:Node {RDF: $rdf})
                    SET n.Title = $info.Title,
                        n.Description = $info.Description, 
                        n.MDI = $info.MDI, 
                        n.URL = $info.URL, 
                        n.Alias = $info.Alias
                    """
                )
                session.run(query, rdf=rdf, info=info)
        driver.close()
    
    def update_node_category(self, rdf_info_map: Dict) -> None:
        driver = self.get_drive()
        with driver.session() as session:
            for rdf, info in tqdm(rdf_info_map.items(), desc='Updating nodes'):
                query = (
                    """
                    MATCH (n:Node {RDF: $rdf})
                    UNWIND $categories_values AS cv
                    SET n.Category = cv.Category,
                        n.has_category = cv.has_category
                    """
                )
                session.run(query, rdf=rdf,
                            categories_values=info
                            )
        driver.close()
        
    #--------------------------------------------------------------------------
    'Functions to Extract Nodes'
    def match_node(self, rdf: str) -> Dict:
        """Returns the information of the lookup node
        MUST BE IN RDF FORMAT: i.e.: Q76 is the RDF code for Barack Obama"""
        assert rdf.startswith('Q'), "rdf must be a vlid RDF identifier starting with 'Q'"
        
        driver = self.get_drive()
        
        result = {}
        with driver.session() as session:
            query = ("MATCH (a:Node {RDF: $rdf})"
                     "RETURN a")
            nodes = session.run(query, rdf=rdf)
            result = nodes.single().data()['a'] if nodes.peek() else {} 
            
        driver.close()
        return result
    
    def match_related_nodes(self, rdf: str, direction: str = 'any',
                            rdf_only: bool = False) -> Tuple[List[any], List[any]]:
        """
        Returns the list of nodes and relationships that are connected to the input node.

        Args:
            rdf (str): The RDF identifier of the node.
            direction (str): The direction of relationships to match. 
                             'any' for undirected, '<-' for directed inwards, '->' for directed outwards.
             rdf_only (bool): If True, returns only the RDF identifiers of the nodes and the properties of the relationships in the path. If False, returns full details of the nodes and relationships (default is False).

        Returns:
            Tuple[List[Dict], List[Dict]]: A tuple containing:
                - A list of nodes connected to the input node.
                - A list of relationships connected to the input node.
        """
        assert rdf.startswith('Q'), "rdf must be a valid RDF identifier starting with 'Q'"
        assert direction in {'any', '<-', '->'}, "direction must be one of 'any', '<-', or '->'"

        driver = self.get_drive()

        nodes, rels = [], []
        try:
            with driver.session() as session:
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
    
    def find_path(self, rdf_start: str, rdf_end: str, min_hops: int = 2, max_hops: int = 3, limit: int = 1,
                  relationship_types: List[str] = None, rdf_only: bool = False) -> List[Tuple[List[any], List[any]]]:
        """
        Finds multiple paths between two nodes identified by RDF, within a specified hop range.
    
        Args:
            rdf_start (str): The RDF identifier for the start node.
            rdf_end (str): The RDF identifier for the end node.
            min_hops (int): Minimum number of hops in the path (default is 2). This sets the minimum length of the path.
            max_hops (int): Maximum number of hops in the path (default is 3). This sets the maximum length of the path.
            limit (int): Max number of paths to generate (default is 1). Controls how many paths to return.
            relationship_types (List[str], optional): A list of relationship types to filter the paths. If provided, only paths containing these relationships will be considered. If not provided, all relationship types are considered.
            rdf_only (bool): If True, returns only the RDF identifiers of the nodes and the properties of the relationships in the path. If False, returns full details of the nodes and relationships (default is False).
    
        Returns:
            List[Tuple[List[any], List[any]]]: A list of tuples, where each tuple contains two lists:
                - The first list contains the nodes in the path, with each node represented as a dictionary (or just the RDF identifier if `rdf_only` is True).
                - The second list contains the relationships in the path, with each relationship represented as a dictionary (or just the relationship property if `rdf_only` is True).
        """
        
        assert rdf_start.startswith('Q'), "rdf_start must be a valid RDF identifier starting with 'Q'"
        assert rdf_end.startswith('Q'), "rdf_end must be a valid RDF identifier starting with 'Q'"
    
        driver = self.get_drive()
        
        paths = []
        try:
            with driver.session() as session:
                # Construct the query with or without relationship types filtering
                relationship_filter = '|'.join(relationship_types) if relationship_types else ''
                relationship_part = f"[r:{relationship_filter}*{min_hops}..{max_hops}]" if relationship_types else f"[*{min_hops}..{max_hops}]"
                
                query = (
                    f"""
                    MATCH path = (n {{RDF: $rdf_start}})-{relationship_part}-(m {{RDF: $rdf_end}})
                    RETURN nodes(path) AS nodes, relationships(path) AS relationships
                    LIMIT $limit
                    """
                )
                        
                result = session.run(query, rdf_start=rdf_start, rdf_end=rdf_end, limit=limit)
                
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