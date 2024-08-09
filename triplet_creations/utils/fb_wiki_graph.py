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
        assert 'Q' in rdf
        
        driver = self.get_drive()
        
        result = {}
        with driver.session() as session:
            query = ("MATCH (a:Node {RDF: $rdf})"
                     "RETURN a")
            nodes = session.run(query, rdf=rdf)
            result = nodes.single().data()['a'] if nodes.peek() else {} 
            
        driver.close()
        return result
    
    def match_connected(self, rdf: str) -> Tuple[List[Dict], List[Dict]]:
        """ Returns the list of nodes and relations that are attached to the input node.
        MUST BE IN RDF FORMAT: i.e.: Q76 is the RDF code for Barack Obama"""
        assert 'Q' in rdf
        
        driver = self.get_drive()
        
        nodes, rels = [], []
        with driver.session() as session:
            query = ("MATCH (n:Node {RDF: $rdf})-[r]-(connected)"
                     "RETURN r, connected")
            nod_rel  = session.run(query, rdf=rdf)
            
            if nod_rel.peek():
                for record in nod_rel:
                    # Extract relationship properties
                    rel_properties = dict(record['r'])
                    rels.append(rel_properties)
                    
                    # Extract node properties
                    node_properties = dict(record['connected'])
                    nodes.append(node_properties)
            
        driver.close()
        return nodes, rels
    
    def match_inwards(self, rdf: str) -> Tuple[List[Dict], List[Dict]]:
        """ Returns the list of nodes and relations that are going into the input node.
        MUST BE IN RDF FORMAT: i.e.: Q76 is the RDF code for Barack Obama"""
        assert 'Q' in rdf
        
        driver = self.get_drive()
        
        nodes, rels = [], []
        with driver.session() as session:
            query = ("MATCH (n:Node {RDF: $rdf})<-[r]-(connected)"
                     "RETURN r, connected")
            nod_rel  = session.run(query, rdf=rdf)
            
            if nod_rel.peek():
                for record in nod_rel:
                    # Extract relationship properties
                    rel_properties = dict(record['r'])
                    rels.append(rel_properties)
                    
                    # Extract node properties
                    node_properties = dict(record['connected'])
                    nodes.append(node_properties)
            
        driver.close()
        return nodes, rels
    
    def match_outwards(self, rdf: str) -> Tuple[List[Dict], List[Dict]]:
        """ Returns the list of nodes and relations that are going out of the input node.
        MUST BE IN RDF FORMAT: i.e.: Q76 is the RDF code for Barack Obama"""
        assert 'Q' in rdf
        
        driver = self.get_drive()
        
        nodes, rels = [], []
        with driver.session() as session:
            query = ("MATCH (n:Node {RDF: $rdf})-[r]->(connected)"
                     "RETURN r, connected")
            nod_rel  = session.run(query, rdf=rdf)
            
            if nod_rel.peek():
                for record in nod_rel:
                    # Extract relationship properties
                    rel_properties = dict(record['r'])
                    rels.append(rel_properties)
                    
                    # Extract node properties
                    node_properties = dict(record['connected'])
                    nodes.append(node_properties)
            
        driver.close()
        return nodes, rels
    
    def find_path(self, rdf_start: str, rdf_end: str, min_hops: int = 2, max_hops: int = 3, limit: int = 1) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Finds multiple paths between two nodes identified by RDF, within a specified hop range.
        
        Args:
            rdf_start (str): The RDF identifier for the start node.
            rdf_end (str): The RDF identifier for the end node.
            min_hops (int): Minimum number of hops in the path (default is 2).
            max_hops (int): Maximum number of hops in the path (default is 3).
            limit (int): Max number of paths to generate (default is 1).
        
        Returns:
            Tuple[List[Dict], List[Dict]]: A tuple containing a list of nodes and relationships in the path.
        """
        assert 'Q' in rdf_start
        assert 'Q' in rdf_end
    
        driver = self.get_drive()
        
        paths = []
        with driver.session() as session:
            query = (
                f"""
                MATCH path = (n {{RDF: $rdf_start}})-[*{min_hops}..{max_hops}]-(m {{RDF: $rdf_end}})
                RETURN nodes(path) AS nodes, relationships(path) AS relationships
                LIMIT $limit
                """
            )
            result = session.run(query, rdf_start=rdf_start, rdf_end=rdf_end, limit=limit)
            
            if result.peek():
                for record in result:
                    nodes = [dict(node) for node in record['nodes']]
                    relationships = [dict(rel) for rel in record['relationships']]
                    paths.append((nodes, relationships))
        
        driver.close()
        return paths