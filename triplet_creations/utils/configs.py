# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:14:54 2024

@author: Eduin Hernandez
"""
from configparser import ConfigParser

def global_configs(config_path: str) -> dict:
    parser = ConfigParser()
    try:
        parser.read(config_path)
        
        neo4j = {'uri': parser.get("Neo4j", "uri"),
                   'user': parser.get("Neo4j", "user"),
                   'password': parser.get("Neo4j", "password")}
        
        configs = {'Neo4j': neo4j}
        return configs
    except:
        assert False, "Error in the configurations!"