TAILS_WITH_QUALIFIERS= """
SELECT
  ?head_uri ?property ?statement
  ?qual_property_uri ?qual_value ?qual_v_is_item
WHERE {{
  ?head_uri ?p_direct ?statement .
  ?statement ?property_statement wd:{entity_id} .
  ?property wikibase:statementProperty ?property_statement .

 OPTIONAL {{
  # pg_uri is the qualifier property. e.g. Country, for Boston tail
  # qual_v_node is the qualifier value. e.g. United States, for Boston tail
  ?statement ?pq_uri ?qual_v_node .
  ?qual_property_uri wikibase:qualifier ?pq_uri .

  OPTIONAL {{
    # Check if the qualifier value node is a wikibase Item (more reliable than wikiPageWikiLink)
    # ?qual_v_node a wikibase:Item . 
    ?qual_v_node wdt:P31 ?instance_of_value . # Another reliable check

    BIND(REPLACE(STR(?qual_v_node), "http://www.wikidata.org/entity/", "") AS ?qual_v_temp) # Extract QID
    BIND(true AS ?qual_v_is_item_temp) # Set to true since it's an Item
  }}
  OPTIONAL {{
    # If qual_v_node is deemed a literal, and the previous check did not pass.
    FILTER(ISLITERAL(?qual_v_node) && !BOUND(?qual_v_temp))
    BIND(STR(?qual_v_node) AS ?qual_v_temp)
    BIND(false AS ?qual_v_is_item_temp)
  }}
  BIND(COALESCE(?qual_v_temp, STR(?qual_v_node)) AS ?qual_value)
  BIND(COALESCE(?qual_v_is_item_temp, false) AS ?qual_v_is_item)
}}

  # Uncomment for labels, but adds overhead. Process labels from IDs later if needed.
  # SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
"""

# TAILS_WITHOUT_QUALIFIERS = """
# SELECT
#   ?head_uri ?property ?statement
# WHERE {{
#   ?head_uri ?p_direct ?statement .
#   ?statement ?property_statement wd:{entity_id} .
#   ?property wikibase:statementProperty ?property_statement .
# }}
# LIMIT {limit}
# """
TAILS_WITHOUT_QUALIFIERS_COMPLICATED = """
SELECT
  ?head_uri ?property ?statement
WHERE {{
  # Find statement nodes where the entity_id is the main value
  ?statement ?property_statement wd:{entity_id} .

  # Link the statement node back to the item (head_uri)
  ?head_uri ?p ?statement .

  # Link the statement property to the main property
  ?property wikibase:statementProperty ?property_statement .

  # Explicitly state the relationship between the main property and the 'p:' property
  ?property wikibase:claim ?p .
}}
LIMIT {limit}
"""
TAILS_WITHOUT_QUALIFIERS= """
SELECT ?subject ?predicate
WHERE {{
 ?subject ?predicate wd:{entity_id}
}}
LIMIT {limit}
"""
